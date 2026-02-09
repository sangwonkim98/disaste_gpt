"""
OCR 처리 모듈
PDF 텍스트 추출 및 캐싱 관리
"""

import os
import logging
import fitz  # PyMuPDF
import re
import unicodedata
from pathlib import Path
from datetime import datetime
from mistralai import Mistral, DocumentURLChunk

from config import MISTRAL_API_KEY, OCR_OUTPUT_DIR

logger = logging.getLogger(__name__)

class OCRProcessor:
    """PDF OCR 처리 및 캐싱 관리"""
    
    def __init__(self, api_key: str = MISTRAL_API_KEY, output_dir: Path = OCR_OUTPUT_DIR):
        self.mistral_client = Mistral(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"OCR 프로세서 초기화: {output_dir}")
    
    def clean_text(self, text: str) -> str:
        """추출된 텍스트에서 이상한 문자들을 정리"""
        if not text:
            return text

        # 한글 NFD → NFC 정규화 (자모 분해 방지)
        text = unicodedata.normalize('NFC', text)

        # LaTeX 수식 정리
        cleaned_text = text
        
        # LaTeX 수식 패턴들 정리
        latex_patterns = [
            (r'\$\s*\\bigcirc\s*\$', '○'),  # $\bigcirc$ -> ○
            (r'\$\s*\\cdot\s*\$', '·'),     # $\cdot$ -> ·
            (r'\$\s*\\times\s*\$', '×'),    # $\times$ -> ×
            (r'\$\s*\\alpha\s*\$', 'α'),    # $\alpha$ -> α
            (r'\$\s*\\beta\s*\$', 'β'),     # $\beta$ -> β
            (r'\$\s*\\gamma\s*\$', 'γ'),    # $\gamma$ -> γ
            (r'\$\s*\\delta\s*\$', 'δ'),    # $\delta$ -> δ
            (r'\$\s*\\epsilon\s*\$', 'ε'),  # $\epsilon$ -> ε
            (r'\$\s*\\lambda\s*\$', 'λ'),   # $\lambda$ -> λ
            (r'\$\s*\\mu\s*\$', 'μ'),       # $\mu$ -> μ
            (r'\$\s*\\pi\s*\$', 'π'),       # $\pi$ -> π
            (r'\$\s*\\sigma\s*\$', 'σ'),    # $\sigma$ -> σ
            (r'\$\s*\\tau\s*\$', 'τ'),      # $\tau$ -> τ
            (r'\$\s*\\phi\s*\$', 'φ'),      # $\phi$ -> φ
            (r'\$\s*\\omega\s*\$', 'ω'),    # $\omega$ -> ω
        ]
        
        for pattern, replacement in latex_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
        # 기타 특수 문자 정리
        special_patterns = [
            (r'※', '※'),  # ※는 그대로 유지 (참조 표시)
            (r'\\', ''),  # 백슬래시 제거
            (r'\s+', ' '),  # 여러 공백을 하나로
        ]
        
        for pattern, replacement in special_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # 줄바꿈 정리
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # 빈 줄 정리
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def get_ocr_output_path(self, pdf_path: str) -> Path:
        """PDF 파일에 대응하는 OCR 결과 파일 경로 반환"""
        pdf_name = Path(pdf_path).stem
        return self.output_dir / f"{pdf_name}_ocr.txt"
    
    def save_ocr_result(self, pdf_path: str, ocr_text: str) -> bool:
        """OCR 결과를 텍스트 파일로 저장"""
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
        """저장된 OCR 결과 로드"""
        try:
            output_path = self.get_ocr_output_path(pdf_path)
            
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 헤더 부분 제거하고 실제 OCR 텍스트만 반환
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
        """Mistral OCR을 사용하여 PDF에서 텍스트 추출"""
        try:
            pdf_file = Path(pdf_path)
            assert pdf_file.is_file()

            logger.info(f"Mistral OCR 시작: {pdf_file.name}")
            
            # PDF 파일 업로드
            uploaded_file = self.mistral_client.files.upload(
                file={
                    "file_name": pdf_file.stem,
                    "content": pdf_file.read_bytes(),
                },
                purpose="ocr",
            )

            # 서명된 URL 획득
            signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

            # OCR 처리
            pdf_response = self.mistral_client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )

            # OCR 결과에서 텍스트 추출
            text = ""
            for page in pdf_response.pages:
                text += page.markdown + "\n"

            logger.info(f"Mistral OCR 완료: {len(text):,}자")
            return text
            
        except Exception as e:
            logger.error(f"Mistral OCR 처리 중 오류 발생: {str(e)}")
            return ""
    
    def extract_pdf_text_fallback(self, pdf_path: str) -> str:
        """PyMuPDF를 사용한 기본 텍스트 추출"""
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
            # 텍스트 정리
            cleaned_text = self.clean_text(full_text)
            logger.info(f"PyMuPDF 완료: {len(full_text):,}자 -> {len(cleaned_text):,}자 (정리 후)")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"PyMuPDF 텍스트 추출 실패: {e}")
            return ""
    
    def extract_pdf_text(self, pdf_path: str, use_cache: bool = True) -> str:
        """PDF 텍스트 추출 (OCR 우선, 폴백 지원)"""
        import time
        start_time = time.time()
        
        pdf_name = Path(pdf_path).name
        logger.info(f"📄 PDF 텍스트 추출 시작: {pdf_name}")
        
        try:
            # 캐시 확인
            if use_cache:
                cache_start = time.time()
                cached_text = self.load_ocr_result(pdf_path)
                cache_time = time.time() - cache_start
                
                if cached_text:
                    # 캐시된 텍스트도 정리
                    cleaned_cached_text = self.clean_text(cached_text)
                    total_time = time.time() - start_time
                    logger.info(f"✅ 캐시된 OCR 결과 사용: {pdf_name} ({len(cached_text):,}자 -> {len(cleaned_cached_text):,}자, {total_time:.2f}초)")
                    return cleaned_cached_text
                
                logger.info(f"📦 캐시 없음 ({cache_time:.2f}초), 새로 처리")
            
            # Mistral OCR 시도
            ocr_start = time.time()
            ocr_text = self.perform_mistral_ocr(pdf_path)
            ocr_time = time.time() - ocr_start
            
            if ocr_text and ocr_text.strip():
                # OCR 성공 - 텍스트 정리 후 저장
                cleaned_ocr_text = self.clean_text(ocr_text)
                if use_cache:
                    self.save_ocr_result(pdf_path, cleaned_ocr_text)
                
                total_time = time.time() - start_time
                logger.info(f"✅ Mistral OCR 성공: {pdf_name} ({len(ocr_text):,}자 -> {len(cleaned_ocr_text):,}자, {total_time:.2f}초)")
                return cleaned_ocr_text
            else:
                # OCR 실패 - 폴백 사용
                logger.warning(f"❌ Mistral OCR 실패, PyMuPDF 폴백 시도")
                fallback_start = time.time()
                fallback_text = self.extract_pdf_text_fallback(pdf_path)
                fallback_time = time.time() - fallback_start
                
                if fallback_text and use_cache:
                    self.save_ocr_result(pdf_path, fallback_text)
                
                total_time = time.time() - start_time
                logger.info(f"✅ PyMuPDF 폴백 완료: {pdf_name} ({len(fallback_text):,}자, {total_time:.2f}초)")
                return fallback_text  # 이미 clean_text 적용됨
                
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ PDF 텍스트 추출 오류 ({total_time:.2f}초): {e}")
            
            # 최후 수단으로 폴백 시도
            return self.extract_pdf_text_fallback(pdf_path)
    
    def get_cached_files(self) -> list:
        """캐시된 OCR 파일 목록 반환"""
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

def test_ocr_processor():
    """OCR 프로세서 테스트"""
    from config import PDF_FILES
    
    processor = OCRProcessor()
    
    print("🧪 OCR 프로세서 테스트")
    
    # 캐시된 파일 확인
    cached_files = processor.get_cached_files()
    print(f"📦 캐시된 파일: {len(cached_files)}개")
    
    # 첫 번째 PDF 테스트 (있는 경우)
    if PDF_FILES and os.path.exists(PDF_FILES[0]):
        test_pdf = PDF_FILES[0]
        print(f"📄 테스트 PDF: {Path(test_pdf).name}")
        
        # 텍스트 추출 테스트
        text = processor.extract_pdf_text(test_pdf)
        print(f"✅ 추출된 텍스트: {len(text):,}자")
        print(f"📝 미리보기: {text[:200]}...")
        
        return True
    else:
        print("❌ 테스트용 PDF 파일이 없습니다.")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ocr_processor()
