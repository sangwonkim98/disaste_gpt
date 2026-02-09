"""
고급 RAG 시스템 (Query Rewriting + Reranking)
 - vLLM 기반 쿼리 재작성 (재시도 로직 포함)
 - Cross-Encoder 전역 재순위화 (멀티 PDF 지원)
 - 문서 임베딩 캐싱 및 재사용
 - 캐시 무효화 전략 (모델/청크 설정 해시 포함)
 - GPU 자원 싱글턴 관리
 - 다양성 제약 (문서당 최대 청크 수)

개선사항:
- Cross-Encoder 전역 랭킹 문제 해결
- 임베딩 재순위화 시 embed_documents() 사용
- 쿼리 재작성 가중치를 max → mean으로 개선
- vLLM 호출 재시도 로직 추가
- 로그 레벨 개선 (민감정보 DEBUG 전용)
"""

import os
import logging
import json
import hashlib
import time
import unicodedata
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# LangChain imports
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

# LangChain reranker
try:
    from langchain_community.document_compressors import FlashrankRerank
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

from config import (
    EMBEDDING_MODEL, EMBEDDING_CACHE_DIR, VECTORSTORE_CACHE_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
    # vLLM 서버 설정
    VLLM_SERVER_URL, VLLM_API_KEY, LLM_MODEL_NAME, TEMPERATURE, TOP_P,
    # GPU 분리 설정
    RAG_GPU_DEVICE, FAISS_MODE
)

from ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)


def _get_config_default(name: str, default_value):
    """config.py에 옵션이 없을 경우를 대비한 안전한 기본값 제공"""
    try:
        import config as _cfg
        return getattr(_cfg, name, default_value)
    except Exception:
        return default_value


# Query Rewriting 기본값
QUERY_REWRITING_ENABLED = _get_config_default("QUERY_REWRITING_ENABLED", True)
QUERY_REWRITE_NUM = _get_config_default("QUERY_REWRITE_NUM", 3)
QUERY_REWRITE_TIMEOUT = _get_config_default("QUERY_REWRITE_TIMEOUT", 25)
QUERY_REWRITE_MAX_RETRIES = _get_config_default("QUERY_REWRITE_MAX_RETRIES", 2)

# Reranking 기본값
RERANKING_ENABLED = _get_config_default("RERANKING_ENABLED", True)
RERANK_TOP_N = _get_config_default("RERANK_TOP_N", 24)
CANDIDATES_PER_QUERY = _get_config_default("CANDIDATES_PER_QUERY", 12)
ORIGINAL_QUERY_WEIGHT = _get_config_default("ORIGINAL_QUERY_WEIGHT", 0.6)
REWRITE_QUERIES_WEIGHT = _get_config_default("REWRITE_QUERIES_WEIGHT", 0.4)

# Cross-Encoder 설정
CROSS_ENCODER_RERANKING_ENABLED = _get_config_default("CROSS_ENCODER_RERANKING_ENABLED", True)
CROSS_ENCODER_MODEL = _get_config_default("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")
CROSS_ENCODER_TOP_N = _get_config_default("CROSS_ENCODER_TOP_N", TOP_K_RESULTS)

# 다양성 제약
MAX_CHUNKS_PER_DOC = _get_config_default("MAX_CHUNKS_PER_DOC", 3)


# GPU 자원 싱글턴
_GPU_RESOURCES = None

def _get_gpu_resources():
    """GPU 자원을 싱글턴으로 관리"""
    global _GPU_RESOURCES
    if _GPU_RESOURCES is None:
        try:
            import faiss
            if faiss.get_num_gpus() > 0:
                _GPU_RESOURCES = faiss.StandardGpuResources()
                _GPU_RESOURCES.setTempMemory(512 * 1024 * 1024)  # 512MB
                logger.info("GPU 자원 싱글턴 초기화 완료")
        except Exception as e:
            logger.warning(f"GPU 자원 초기화 실패: {e}")
    return _GPU_RESOURCES


def _get_cache_hash() -> str:
    """캐시 무효화를 위한 설정 해시 생성"""
    config_str = f"{EMBEDDING_MODEL}_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


class AdvancedRAGSystem:
    """Query rewriting + reranking 지원 RAG 시스템 (개선 버전)"""

    def __init__(self):
        # FAISS 모드 설정 (config에서 로드)
        # FAISS_MODE="cpu"면 GPU 사용 안 함 (OOM 방지, 권장)
        # FAISS_MODE="gpu"면 GPU 사용
        self.faiss_mode = FAISS_MODE.lower()

        if self.faiss_mode == "gpu":
            self.use_gpu = self._check_gpu_availability()
            if self.use_gpu:
                logger.info("GPU FAISS 사용 가능 - GPU 가속 활성화")
            else:
                logger.info("GPU FAISS 사용 불가 - CPU 모드로 폴백")
        else:
            self.use_gpu = False
            logger.info(f"FAISS_MODE={FAISS_MODE} - CPU 모드로 실행 (OOM 방지)")

        # 디바이스 설정 (Embedding용 - vLLM과 분리된 GPU 사용)
        import torch
        if torch.cuda.is_available() and RAG_GPU_DEVICE:
            # RAG 전용 GPU 사용 (GPU 2)
            gpu_id = int(RAG_GPU_DEVICE)
            if gpu_id < torch.cuda.device_count():
                self.device = f"cuda:{gpu_id}"
                logger.info(f"🔥 RAG 전용 GPU 사용: {self.device} (vLLM과 분리)")
            else:
                self.device = "cpu"
                logger.warning(f"GPU {gpu_id} 없음, CPU 모드로 실행")
        else:
            self.device = "cpu"
            logger.info("💻 CPU 모드로 실행")

        # 임베딩 로더
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            cache_folder=str(EMBEDDING_CACHE_DIR),
            model_kwargs={"device": self.device}
        )

        # OCR
        self.ocr_processor = OCRProcessor()

        # VectorStore 캐시
        self.pdf_vectorstores: Dict[str, FAISS] = {}
        self.pdf_documents: Dict[str, List[Document]] = {}
        self.pdf_names: Dict[str, str] = {}
        
        # 문서 임베딩 캐시 (재순위화 성능 개선)
        self.pdf_embeddings_cache: Dict[str, np.ndarray] = {}

        # 청킹
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            keep_separator=True
        )

        # 캐시 디렉토리 (해시 포함)
        self.cache_hash = _get_cache_hash()
        self.vectorstore_cache_dir = Path(VECTORSTORE_CACHE_DIR) / f"v_{self.cache_hash}"
        self.vectorstore_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📦 캐시 디렉토리: {self.vectorstore_cache_dir} (해시: {self.cache_hash})")

        # vLLM endpoint
        self.vllm_endpoint = VLLM_SERVER_URL.rstrip("/") + "/chat/completions"
        self.vllm_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLLM_API_KEY}" if VLLM_API_KEY else ""
        }

        # LangChain Reranker (FlashrankRerank)
        self.reranker = None
        if CROSS_ENCODER_RERANKING_ENABLED and RERANKER_AVAILABLE:
            try:
                logger.info(f"🔧 Reranker 초기화: FlashrankRerank (top_n={CROSS_ENCODER_TOP_N})")
                self.reranker = FlashrankRerank(
                    top_n=CROSS_ENCODER_TOP_N,
                    model="ms-marco-MiniLM-L-12-v2"  # Flashrank 기본 모델
                )
                logger.info(f"✅ Reranker 로드 완료 (top_n: {CROSS_ENCODER_TOP_N})")
            except Exception as e:
                logger.warning(f"Reranker 로드 실패, 임베딩 기반 재순위화로 폴백: {e}")
                self.reranker = None
        elif CROSS_ENCODER_RERANKING_ENABLED and not RERANKER_AVAILABLE:
            logger.warning("FlashrankRerank 사용 불가. 임베딩 기반 재순위화 사용")
            self.reranker = None

        logger.info("🚀 Advanced RAG 시스템 초기화 완료")

    def _check_gpu_availability(self) -> bool:
        """GPU 사용 가능 여부 확인 (싱글턴 자원 활용)"""
        try:
            import faiss
            gpu_count = faiss.get_num_gpus()
            if gpu_count > 0:
                # 싱글턴 GPU 자원 초기화 시도
                gpu_res = _get_gpu_resources()
                return gpu_res is not None
            return False
        except Exception as e:
            logger.warning(f"GPU FAISS 확인 실패: {e}")
            return False

    def _create_faiss_index(self, embeddings: np.ndarray, use_gpu: bool = None) -> "faiss.Index":
        """FAISS 인덱스 생성 (GPU 싱글턴 자원 활용)"""
        import faiss
        if use_gpu is None:
            use_gpu = self.use_gpu

        dimension = embeddings.shape[1]
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        if use_gpu:
            try:
                cpu_index = faiss.IndexFlatIP(dimension)
                gpu_res = _get_gpu_resources()
                if gpu_res is not None:
                    # RAG_GPU_DEVICE는 프로세스 내부 인덱스 (CUDA_VISIBLE_DEVICES 적용 후)
                    gpu_id = int(RAG_GPU_DEVICE) if RAG_GPU_DEVICE else 0
                    gpu_index = faiss.index_cpu_to_gpu(gpu_res, gpu_id, cpu_index)
                    gpu_index.add(embeddings_norm.astype("float32"))
                    return gpu_index
                else:
                    logger.warning("GPU 자원 없음, CPU로 폴백")
            except Exception as e:
                logger.warning(f"GPU FAISS 생성 실패, CPU로 폴백: {e}")

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_norm.astype("float32"))
        return index

    def _save_vectorstore_with_gpu_support(self, vectorstore: FAISS, cache_path: Path) -> bool:
        try:
            import faiss
            index_type = type(vectorstore.index).__name__
            is_gpu_index = "Gpu" in index_type or hasattr(vectorstore.index, "getDevice")
            if is_gpu_index:
                cpu_index = faiss.index_gpu_to_cpu(vectorstore.index)
                original_index = vectorstore.index
                vectorstore.index = cpu_index
                vectorstore.save_local(str(cache_path))
                vectorstore.index = original_index
            else:
                vectorstore.save_local(str(cache_path))
            return True
        except Exception as e:
            logger.warning(f"벡터스토어 캐시 저장 실패: {e}")
            return False

    def _load_vectorstore_with_gpu_support(self, cache_path: Path, pdf_path: str) -> Optional[FAISS]:
        try:
            import faiss
            vectorstore = FAISS.load_local(
                str(cache_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(512 * 1024 * 1024)
                    gpu_id = int(RAG_GPU_DEVICE) if RAG_GPU_DEVICE else 0
                    gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, vectorstore.index)
                    vectorstore.index = gpu_index
                except Exception as e:
                    logger.warning(f"GPU 복사 실패, CPU 모드 사용: {e}")
            self.pdf_vectorstores[pdf_path] = vectorstore
            return vectorstore
        except Exception as e:
            logger.warning(f"캐시 로드 실패: {e}")
            return None

    # =====================
    # Query Rewriting (vLLM with Retry)
    # =====================
    def _rewrite_queries(self, user_query: str) -> List[str]:
        """쿼리 재작성 (재시도 로직 포함)"""
        if not QUERY_REWRITING_ENABLED or QUERY_REWRITE_NUM <= 0:
            return []

        import requests
        from time import sleep
        
        system_prompt = (
            "당신은 검색 전문가입니다. 사용자의 한국어 질문을 정보검색에 최적화된 짧은 질의로 다양한 관점에서 재작성하세요. "
            "- 도메인 용어/동의어/축약/완곡 표현 혼합\n"
            "- 불필요한 수식어 제거\n"
            "- 1줄당 1개의 재작성 쿼리\n"
            "- 질문의 의도를 벗어나지 않기"
        )
        user_prompt = (
            f"원본 질문: {user_query}\n"
            f"위 질문을 서로 다른 관점으로 {QUERY_REWRITE_NUM}개 재작성하세요."
        )

        payload = {
            "model": LLM_MODEL_NAME,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # Query Rewriting에는 reasoning 불필요 - 빠른 응답을 위해 비활성화
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        }

        # 재시도 로직 (지수 백오프)
        for attempt in range(QUERY_REWRITE_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    self.vllm_endpoint, 
                    headers=self.vllm_headers, 
                    json=payload, 
                    timeout=QUERY_REWRITE_TIMEOUT
                )
                resp.raise_for_status()
                data = resp.json()
                message = data.get("choices", [{}])[0].get("message", {})
                
                # EXAONE reasoning mode: content 또는 reasoning_content 사용
                content = message.get("content") or message.get("reasoning_content")
                
                # content가 None이거나 비어있는 경우 처리
                if not content:
                    logger.warning(f"vLLM API에서 빈 응답 반환 (시도 {attempt+1})")
                    if attempt < QUERY_REWRITE_MAX_RETRIES:
                        wait_time = (2 ** attempt) * 0.5
                        sleep(wait_time)
                        continue
                    else:
                        return []
                
                # DEBUG 레벨에서만 재작성 쿼리 출력 (민감정보 보호)
                logger.debug(f"재작성 쿼리 응답: {content}")
                
                lines = [l.strip("- •\t ") for l in content.splitlines() if l.strip()]
                rewrites: List[str] = []
                for line in lines:
                    # 단순 번호 제거
                    cleaned = line
                    if ":" in cleaned and cleaned.split(":")[0].strip().isdigit():
                        cleaned = cleaned.split(":", 1)[1].strip()
                    elif len(cleaned) > 2 and cleaned[:2].isdigit() and cleaned[1] == ".":
                        cleaned = cleaned[2:].strip()
                    if cleaned:
                        rewrites.append(cleaned)
                    if len(rewrites) >= QUERY_REWRITE_NUM:
                        break
                
                # 재작성 쿼리 내용 출력 (검색 품질 확인용)
                logger.info(f"✅ 쿼리 재작성 성공: {len(rewrites)}개 생성")
                for i, rq in enumerate(rewrites, 1):
                    logger.info(f"   {i}. {rq}")
                return rewrites
                
            except Exception as e:
                if attempt < QUERY_REWRITE_MAX_RETRIES:
                    wait_time = (2 ** attempt) * 0.5  # 지수 백오프: 0.5s, 1s, 2s...
                    logger.warning(f"쿼리 재작성 실패 (시도 {attempt+1}/{QUERY_REWRITE_MAX_RETRIES+1}), {wait_time}초 후 재시도: {e}")
                    sleep(wait_time)
                else:
                    logger.warning(f"쿼리 재작성 최종 실패, 원문만 사용: {e}")
                    return []
        
        return []

    # =====================
    # Reranking helpers (개선 버전)
    # =====================
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _rerank_with_embeddings(self, query: str, candidate_docs: List[Document], rewrite_queries: List[str]) -> List[Tuple[Document, float]]:
        """임베딩 기반 재순위화 (embed_documents() 사용, mean 가중치)"""
        if not candidate_docs:
            return []
        
        # 쿼리 임베딩
        original_query_emb = self.embeddings.embed_query(query)
        rewrite_embs = [self.embeddings.embed_query(q) for q in rewrite_queries] if rewrite_queries else []

        # 문서 임베딩 (embed_documents() 사용 - 올바른 분포)
        doc_texts = [doc.page_content for doc in candidate_docs]
        doc_embs = self.embeddings.embed_documents(doc_texts)

        scored: List[Tuple[Document, float]] = []
        for doc, doc_emb in zip(candidate_docs, doc_embs):
            # 원본 쿼리 점수
            score_main = self._cosine_similarity(original_query_emb, np.array(doc_emb))
            
            # 재작성 쿼리들의 평균 점수 (max → mean으로 개선)
            if rewrite_embs:
                rewrite_scores = [self._cosine_similarity(np.array(re), np.array(doc_emb)) for re in rewrite_embs]
                score_rewrites = float(np.mean(rewrite_scores))
            else:
                score_rewrites = 0.0
            
            # 최종 점수 계산
            final_score = ORIGINAL_QUERY_WEIGHT * score_main + REWRITE_QUERIES_WEIGHT * score_rewrites
            scored.append((doc, final_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _rerank_with_langchain(self, query: str, candidate_docs: List[Document]) -> List[Tuple[Document, float]]:
        """LangChain Reranker로 재순위화 (점수 포함 반환)"""
        if not self.reranker:
            return [(doc, 0.0) for doc in candidate_docs]
        try:
            # LangChain reranker로 재순위화
            compressed = self.reranker.compress_documents(candidate_docs, query)
            
            # 점수 추출 (LangChain이 relevance_score를 metadata에 저장)
            scored_docs: List[Tuple[Document, float]] = []
            for i, doc in enumerate(compressed):
                # FlashrankRerank는 relevance_score를 metadata에 저장
                score = doc.metadata.get('relevance_score', 1.0 - (i * 0.1))  # 순서 기반 폴백
                scored_docs.append((doc, float(score)))
            
            return scored_docs
        except Exception as e:
            logger.warning(f"Reranker 재순위화 실패, 원본 순서 유지: {e}")
            return [(doc, 0.0) for doc in candidate_docs]
    
    def _apply_diversity_constraint(self, scored_docs: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """다양성 제약: 한 문서(PDF)당 최대 청크 수 제한"""
        if MAX_CHUNKS_PER_DOC <= 0:
            return scored_docs
        
        doc_chunk_count: Dict[str, int] = {}
        filtered: List[Tuple[Document, float]] = []
        
        for doc, score in scored_docs:
            source = doc.metadata.get("source", "unknown")
            count = doc_chunk_count.get(source, 0)
            
            if count < MAX_CHUNKS_PER_DOC:
                filtered.append((doc, score))
                doc_chunk_count[source] = count + 1
        
        return filtered

    # =====================
    # Document & Vectorstore
    # =====================
    def create_documents_from_pdf(self, pdf_path: str) -> List[Document]:
        import time
        start = time.time()
        text = self.ocr_processor.extract_pdf_text(pdf_path)
        if not text or not text.strip():
            logger.warning(f"텍스트 추출 실패: {Path(pdf_path).name}")
            return []
        # 한글 NFD → NFC 정규화 (자모 분해 방지 안전장치)
        text = unicodedata.normalize('NFC', text)
        chunks = self.text_splitter.split_text(text)
        documents: List[Document] = []
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": pdf_path,
                    "pdf_name": Path(pdf_path).name,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_length": len(chunk),
                }
            ))
        logger.info(f"문서 청킹 완료: {len(documents)}개, {time.time()-start:.2f}s")
        return documents

    def get_vectorstore_cache_path(self, pdf_path: str) -> Path:
        pdf_name = Path(pdf_path).stem
        return self.vectorstore_cache_dir / f"{pdf_name}_vectorstore"

    def build_vectorstore(self, pdf_path: str) -> Optional[FAISS]:
        import time
        start = time.time()
        self.pdf_names[pdf_path] = Path(pdf_path).stem
        cache_path = self.get_vectorstore_cache_path(pdf_path)

        if cache_path.exists():
            vs = self._load_vectorstore_with_gpu_support(cache_path, pdf_path)
            if vs is not None:
                logger.info(f"캐시 로드 완료: {Path(pdf_path).name} ({time.time()-start:.2f}s)")
                return vs

        docs = self.create_documents_from_pdf(pdf_path)
        if not docs:
            return None

        texts = [d.page_content for d in docs]
        embeddings_np = np.array(self.embeddings.embed_documents(texts))
        index = self._create_faiss_index(embeddings_np)

        vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore({i: doc for i, doc in enumerate(docs)}),
            index_to_docstore_id={i: i for i in range(len(docs))}
        )

        self._save_vectorstore_with_gpu_support(vectorstore, cache_path)
        self.pdf_vectorstores[pdf_path] = vectorstore
        self.pdf_documents[pdf_path] = docs
        logger.info(f"벡터스토어 구축 완료: {Path(pdf_path).name} ({time.time()-start:.2f}s)")
        return vectorstore

    def build_index(self, pdf_paths: List[str]):
        for p in pdf_paths:
            try:
                self.build_vectorstore(p)
            except Exception as e:
                logger.warning(f"인덱스 구축 실패({Path(p).name}): {e}")

    # =====================
    # Search (Rewriting + Reranking)
    # =====================
    def _collect_candidates(self, vectorstore: FAISS, query: str, rewrite_queries: List[str], candidate_k: int) -> List[Document]:
        # 기본 후보 수집: 원문 + 재작성 질의
        search_docs = vectorstore.similarity_search(query, k=candidate_k)
        for rq in rewrite_queries:
            try:
                search_docs.extend(vectorstore.similarity_search(rq, k=min(CANDIDATES_PER_QUERY, candidate_k)))
            except Exception as e:
                logger.warning(f"재작성 질의 검색 실패({rq}): {e}")
        # 중복 제거 (source, chunk_id)
        unique = {}
        for d in search_docs:
            key = (d.metadata.get("source"), d.metadata.get("chunk_id"))
            if key not in unique:
                unique[key] = d
        return list(unique.values())

    def search(self, query: str, selected_pdf_path: str = None, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """검색 (Cross-Encoder 전역 랭킹 지원, 다양성 제약 적용)"""
        start = time.time()
        rewrite_queries = self._rewrite_queries(query)
        results: List[Dict] = []

        def _docs_to_results(scored_docs: List[Tuple[Document, float]], use_ce_score: bool = False) -> List[Dict]:
            out: List[Dict] = []
            for d, s in scored_docs:
                out.append({
                    "text": d.page_content,
                    "similarity_score": float(s),
                    "pdf_name": d.metadata.get("pdf_name", "Unknown"),
                    "source": d.metadata.get("source", ""),
                    "chunk_id": d.metadata.get("chunk_id", 0),
                    "metadata": d.metadata,
                    "reranked": use_ce_score,  # CE 재순위화 여부 표시
                })
            return out

        if selected_pdf_path and selected_pdf_path != "all":
            # 단일 PDF 검색
            matched_path = None
            for path in self.pdf_vectorstores.keys():
                if Path(path).stem == Path(selected_pdf_path).stem or Path(path).name == selected_pdf_path:
                    matched_path = path
                    break
            if not matched_path:
                logger.warning(f"선택된 PDF 벡터스토어 없음: {selected_pdf_path}")
                return []

            vs = self.pdf_vectorstores[matched_path]
            candidate_k = max(RERANK_TOP_N if RERANKING_ENABLED else top_k, top_k)
            candidates = self._collect_candidates(vs, query, rewrite_queries, candidate_k)

            if self.reranker and RERANKING_ENABLED:
                scored = self._rerank_with_langchain(query, candidates)
                # 다양성 제약 적용
                scored = self._apply_diversity_constraint(scored)
                results = _docs_to_results(scored[:top_k], use_ce_score=True)
            elif RERANKING_ENABLED:
                scored = self._rerank_with_embeddings(query, candidates, rewrite_queries)
                # 다양성 제약 적용
                scored = self._apply_diversity_constraint(scored)
                results = _docs_to_results(scored[:top_k])
            else:
                scored = [(d, 0.0) for d in candidates]
                results = _docs_to_results(scored[:top_k])

            logger.info(f"검색 완료(단일): {len(results)}개, {time.time()-start:.3f}s")
            logger.info("📋 최종 검색 결과:")
            for i, r in enumerate(results, 1):
                preview = r['text'][:100].replace('\n', ' ')
                logger.info(f"   {i}. [{r['similarity_score']:.4f}] {r['pdf_name']} (청크 #{r['chunk_id']}) - {preview}...")
            return results

        # 전체 PDF 검색 - Cross-Encoder 전역 랭킹 문제 해결
        logger.info(f"🔍 전체 PDF 검색 시작 (PDF 수: {len(self.pdf_vectorstores)})")
        
        # 1단계: 모든 PDF에서 후보 수집
        all_candidates: List[Document] = []
        for pdf_path, vs in self.pdf_vectorstores.items():
            try:
                candidate_k = CANDIDATES_PER_QUERY if RERANKING_ENABLED else top_k
                candidates = self._collect_candidates(vs, query, rewrite_queries, candidate_k)
                all_candidates.extend(candidates)
                logger.debug(f"  - {Path(pdf_path).name}: {len(candidates)}개 후보")
            except Exception as e:
                logger.warning(f"검색 실패({Path(pdf_path).name}): {e}")
        
        logger.info(f"📦 총 후보 수: {len(all_candidates)}개")
        
        if not all_candidates:
            logger.warning("후보 문서 없음")
            return []
        
        # 2단계: 전역 재순위화 (모든 후보를 한 번에 처리)
        if self.reranker and RERANKING_ENABLED:
            # LangChain Reranker로 전역 재순위화
            logger.info(f"🔧 Reranker 전역 재순위화 시작 ({len(all_candidates)}개 → {CROSS_ENCODER_TOP_N}개)")
            scored = self._rerank_with_langchain(query, all_candidates)
            # 다양성 제약 적용
            scored = self._apply_diversity_constraint(scored)
            results = _docs_to_results(scored[:top_k], use_ce_score=True)
            
        elif RERANKING_ENABLED:
            # 임베딩 기반 재순위화
            logger.info(f"📊 임베딩 기반 재순위화 시작 ({len(all_candidates)}개)")
            scored = self._rerank_with_embeddings(query, all_candidates, rewrite_queries)
            # 다양성 제약 적용
            scored = self._apply_diversity_constraint(scored)
            results = _docs_to_results(scored[:top_k])
            
        else:
            # 재순위화 없음
            scored = [(d, 0.0) for d in all_candidates]
            results = _docs_to_results(scored[:top_k])

        # 검색 결과 상세 로깅 (품질 확인용)
        logger.info(f"✅ 검색 완료(전체): {len(results)}개, {time.time()-start:.3f}s")
        logger.info("📋 최종 검색 결과:")
        for i, r in enumerate(results, 1):
            preview = r['text'][:100].replace('\n', ' ')
            logger.info(f"   {i}. [{r['similarity_score']:.4f}] {r['pdf_name']} (청크 #{r['chunk_id']}) - {preview}...")
        
        return results

    # =====================
    # Utils (개선 버전)
    # =====================
    def format_search_results(self, results: List[Dict], selected_pdf_path: str = None) -> str:
        """검색 결과 포맷팅 (재순위화 정보 포함)"""
        if not results:
            return "관련된 내용을 찾을 수 없습니다."
        
        if selected_pdf_path and selected_pdf_path != "all":
            pdf_name = Path(selected_pdf_path).stem
            formatted = f"## 📋 관련 문서 내용 (검색 대상: {pdf_name})\n\n"
        else:
            formatted = "## 📋 관련 문서 내용 (전체 문서 검색)\n\n"
        
        for i, r in enumerate(results, 1):
            # 재순위화 뱃지
            rerank_badge = "🔧 Reranked" if r.get('reranked', False) else ""
            
            formatted += f"### {i}. {r['pdf_name']} {rerank_badge}\n"
            formatted += f"**청크 ID:** {r['chunk_id']} | **관련도:** {r['similarity_score']:.4f}\n\n"
            formatted += f"**내용:**\n```\n{r['text']}\n```\n\n"
            formatted += "---\n\n"
        
        return formatted

    def get_pdf_list(self) -> List[Tuple[str, str]]:
        """PDF 목록 반환 (파일 정보 포함)"""
        pdf_list = [("전체 문서에서 검색 (All Documents)", "all")]
        for pdf_path in self.pdf_names.keys():
            pdf_name = self.pdf_names[pdf_path]
            # 벡터스토어 정보
            if pdf_path in self.pdf_vectorstores:
                vs = self.pdf_vectorstores[pdf_path]
                chunk_count = vs.index.ntotal if hasattr(vs.index, 'ntotal') else 0
                display_name = f"{pdf_name} ({chunk_count} chunks)"
            else:
                display_name = pdf_name
            pdf_list.append((display_name, pdf_path))
        return pdf_list

    def get_system_stats(self) -> Dict:
        """시스템 통계 (캐시 해시 포함)"""
        reranker_info = "disabled"
        if self.reranker:
            reranker_info = "flashrank_rerank"
        elif RERANKING_ENABLED:
            reranker_info = "embedding_based"
        
        return {
            "total_pdfs": len(self.pdf_vectorstores),
            "total_vectors": sum(vs.index.ntotal for vs in self.pdf_vectorstores.values()) if self.pdf_vectorstores else 0,
            "gpu_enabled": self.use_gpu,
            "device": self.device,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "cache_hash": self.cache_hash,
            "reranker": reranker_info,
            "query_rewriting": QUERY_REWRITING_ENABLED,
            "max_chunks_per_doc": MAX_CHUNKS_PER_DOC,
        }


def test_advanced_rag():
    from config import PDF_FILES
    rag = AdvancedRAGSystem()
    existing = [p for p in PDF_FILES if Path(p).exists()]
    if not existing:
        print("❌ 테스트용 PDF 파일이 없습니다.")
        return False
    rag.build_vectorstore(existing[0])
    results = rag.search("사업비 집행 기준", existing[0], top_k=3)
    print(f"🔍 결과 {len(results)}개")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['similarity_score']:.3f} | {r['pdf_name']} #{r['chunk_id']}")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_advanced_rag()


