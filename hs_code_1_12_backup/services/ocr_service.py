"""
[OCR Service]
PDF 문서 텍스트 추출 및 정제 담당
Mistral OCR (Vision LLM)을 주력으로 사용하고, 실패 시 PyMuPDF로 폴백하는 하이브리드 방식 사용
"""

import os
import logging
import fitz  # PyMuPDF
import re
from pathlib import Path
from datetime import datetime
from mistralai import Mistral, DocumentURLChunk

from config import MISTRAL_API_KEY, OCR_OUTPUT_DIR

logger = logging.getLogger(__name__)

class OCRProcessor:
    """PDF OCR 처리 및 캐싱 관리 클래스"""
    
    def __init__(self, api_key: str = MISTRAL_API_KEY, output_dir: Path = OCR_OUTPUT_DIR):
        self.mistral_client = Mistral(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"OCR 프로세서 초기화: {output_dir}")
    
    def clean_text(self, text: str) -> str:
        """
        [전처리] 추출된 텍스트에서 불필요한 노이즈 제거
        - LaTeX 수식 문법을 사람이 읽기 편한 기호로 변환
        - 중복 공백, 특수 문자 정리
        """
        if not text:
            return text
        
        # LaTeX 수식 정리
        cleaned_text = text
        
        # LaTeX 수식 패턴들 정리 (예: $\bigcirc$ -> ○)
        latex_patterns = [
            (r'\$\s*\\bigcirc\s*\$', '○'),
            (r'\$\s*\\cdot\s*\$', '·'),
            (r'\$\s*\\times\s*\$', '×'),
            (r'\$\s*\\alpha\s*\$', 'α'),
            (r'\$\s*\\beta\s*\$', 'β'),
            (r'\$\s*\\gamma\s*\$', 'γ'),
            (r'\$\s*\\delta\s*\$', 'δ'),
            (r'\$\s*\\epsilon\s*\$', 'ε'),
            (r'\$\s*\\lambda\s*\$', 'λ'),
            (r'\$\s*\\mu\s*\$', 'μ'),
            (r'\$\s*\\pi\s*\$', 'π'),
            (r'\$\s*\\sigma\s*\$', 'σ'),
            (r'\$\s*\\tau\s*\$', 'τ'),
            (r'\$\s*\\phi\s*\$', 'φ'),
            (r'\$\s*\\omega\s*\$', 'ω'),
        ]
        
        for pattern, replacement in latex_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
        # 기타 특수 문자 정리
        special_patterns = [
            (r'※', '※'),  # ※는 그대로 유지 (참조 표시)
            (r'\\', ''),  # 백슬래시 제거
            (r'\s+', ' '),  # 여러 공백을 하나로 압축
        ]
        
        for pattern, replacement in special_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # 가독성을 위해 단락 구분(줄바꿈 2번) 정리
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def get_ocr_output_path(self, pdf_path: str) -> Path:
        """캐싱 파일 경로 생성 (파일명_ocr.txt)"""
        pdf_name = Path(pdf_path).stem
        return self.output_dir / f"{pdf_name}_ocr.txt"
    
    def save_ocr_result(self, pdf_path: str, ocr_text: str) -> bool:
        """OCR 결과를 로컬 파일로 저장 (캐싱)"""
        try:
            output_path = self.get_ocr_output_path(pdf_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# OCR 결과 - {os.path.basename(pdf_path)}\n")
                f.write(f"# 생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 원본 파일: {pdf_path}\n\n")
                f.write(ocr_text)
            
            logger.info(f"OCR 결과 저장: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"OCR 결과 저장 실패: {e}")
            return False
    
    def load_ocr_result(self, pdf_path: str) -> str:
        """저장된 OCR 결과가 있다면 로드 (캐시 히트)"""
        try:
            output_path = self.get_ocr_output_path(pdf_path)
            
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 상단 헤더(#) 부분 제거하고 실제 본문만 반환
                    lines = content.split('\n')
                    content_start = 0
                    for i, line in enumerate(lines):
                        if line.strip() and not line.startswith('#'):
                            content_start = i
                            break
                    
                    return '\n'.join(lines[content_start:]).strip()
            
            return None
            
        except Exception as e:
            logger.error(f"OCR 결과 로드 실패: {e}")
            return None
    
    def perform_mistral_ocr(self, pdf_path: str) -> str:
        """
        [Primary Strategy] Mistral Vision OCR 사용
        - 장점: 복잡한 레이아웃, 표, 다단 편집 등을 구조적으로(Markdown) 잘 인식함
        - 단점: API 호출 비용 및 네트워크 의존성
        """
        try:
            pdf_file = Path(pdf_path)
            assert pdf_file.is_file()

            logger.info(f"Mistral OCR 시작: {pdf_file.name}")
            
            # 1. PDF 파일 업로드
            uploaded_file = self.mistral_client.files.upload(
                file={
                    "file_name": pdf_file.stem,
                    "content": pdf_file.read_bytes(),
                },
                purpose="ocr",
            )

            # 2. 보안 URL(Signed URL) 발급
            signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

            # 3. OCR 요청 (이미지도 포함하여 분석)
            pdf_response = self.mistral_client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )

            # 4. 결과 조합 (페이지별 Markdown 텍스트 합치기)
            text = ""
            for page in pdf_response.pages:
                text += page.markdown + "\n"

            logger.info(f"Mistral OCR 완료: {len(text):,}자")
            return text
            
        except Exception as e:
            logger.error(f"Mistral OCR 처리 중 오류 발생: {str(e)}")
            return ""
    
    def extract_pdf_text_fallback(self, pdf_path: str) -> str:
        """
        [Secondary Strategy] PyMuPDF (fitz) 사용
        - Mistral API 실패 시 로컬 라이브러리로 즉시 전환
        - 텍스트 레이어가 살아있는 PDF에 대해 빠르고 비용이 들지 않음
        """
        try:
            logger.info(f"PyMuPDF 폴백 처리: {Path(pdf_path).name}")
            
            doc = fitz.open(pdf_path)
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    full_text += f"\n--- 페이지 {page_num + 1} ---\n"
                    full_text += text + "\n"
            
            doc.close()
            
            # 텍스트 정제 적용
            cleaned_text = self.clean_text(full_text)
            logger.info(f"PyMuPDF 완료: {len(full_text):,}자 -> {len(cleaned_text):,}자 (정리 후)")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"PyMuPDF 텍스트 추출 실패: {e}")
            return ""
    
    def extract_pdf_text(self, pdf_path: str, use_cache: bool = True) -> str:
        """
        [Main Interface] 외부에서 호출하는 메인 함수
        1. 캐시 확인 -> 있으면 반환
        2. Mistral OCR 시도 -> 성공 시 저장 후 반환
        3. 실패 시 PyMuPDF 폴백 시도 -> 성공 시 저장 후 반환
        """
        import time
        start_time = time.time()
        
        pdf_name = Path(pdf_path).name
        logger.info(f"📄 PDF 텍스트 추출 시작: {pdf_name}")
        
        try:
            # 1. 캐시 확인
            if use_cache:
                cache_start = time.time()
                cached_text = self.load_ocr_result(pdf_path)
                cache_time = time.time() - cache_start
                
                if cached_text:
                    cleaned_cached_text = self.clean_text(cached_text)
                    total_time = time.time() - start_time
                    logger.info(f"✅ 캐시된 OCR 결과 사용: {pdf_name} ({len(cached_text):,}자, {total_time:.2f}초)")
                    return cleaned_cached_text
                
                logger.info(f"📦 캐시 없음 ({cache_time:.2f}초), 새로 처리")
            
            # 2. Mistral OCR 시도 (Primary)
            ocr_start = time.time()
            ocr_text = self.perform_mistral_ocr(pdf_path)
            ocr_time = time.time() - ocr_start
            
            if ocr_text and ocr_text.strip():
                # 성공 시 정제 및 캐싱
                cleaned_ocr_text = self.clean_text(ocr_text)
                if use_cache:
                    self.save_ocr_result(pdf_path, cleaned_ocr_text)
                
                total_time = time.time() - start_time
                logger.info(f"✅ Mistral OCR 성공: {pdf_name} ({len(ocr_text):,}자 -> {len(cleaned_ocr_text):,}자, {total_time:.2f}초)")
                return cleaned_ocr_text
            else:
                # 3. OCR 실패 -> PyMuPDF 폴백 (Secondary)
                logger.warning(f"❌ Mistral OCR 실패, PyMuPDF 폴백 시도")
                fallback_start = time.time()
                fallback_text = self.extract_pdf_text_fallback(pdf_path)
                fallback_time = time.time() - fallback_start
                
                if fallback_text and use_cache:
                    self.save_ocr_result(pdf_path, fallback_text)
                
                total_time = time.time() - start_time
                logger.info(f"✅ PyMuPDF 폴백 완료: {pdf_name} ({len(fallback_text):,}자, {total_time:.2f}초)")
                return fallback_text
                
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ PDF 텍스트 추출 오류 ({total_time:.2f}초): {e}")
            
            # 최후의 수단: 에러 발생 시에도 로컬 추출 시도
            return self.extract_pdf_text_fallback(pdf_path)
    
    def get_cached_files(self) -> list:
        """디버깅용: 현재 캐시된 파일 목록 반환"""
        try:
            cached_files = []
            for file_path in self.output_dir.glob("*_ocr.txt"):
                cached_files.append({
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime)
                })
            return cached_files
        except Exception as e:
            logger.error(f"캐시 파일 목록 조회 실패: {e}")
            return []

# ... (테스트 코드 생략) ...