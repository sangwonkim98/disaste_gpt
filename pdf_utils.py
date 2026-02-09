"""
PDF 유틸리티 모듈
PDF 뷰어 기능, 페이지 이미지 생성 등
"""

import os
import logging
import tempfile
import fitz  # PyMuPDF
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFUtils:
    """PDF 관련 유틸리티 함수들"""
    
    @staticmethod
    def get_pdf_page_image(pdf_path: str, page_num: int) -> str:
        """PDF 페이지를 이미지로 변환하여 임시 파일 경로 반환"""
        try:
            logger.info(f"PDF 페이지 이미지 생성: {Path(pdf_path).name} 페이지 {page_num + 1}")
            
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            
            # 페이지를 고해상도 이미지로 변환
            mat = fitz.Matrix(2, 2)  # 2배 확대
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(img_data)
                temp_file_path = temp_file.name
            
            doc.close()
            logger.info(f"✅ PDF 페이지 이미지 생성 완료: {temp_file_path}")
            return temp_file_path
            
        except Exception as e:
            logger.error(f"❌ PDF 페이지 이미지 생성 오류: {e}")
            return None
    
    @staticmethod
    def get_pdf_total_pages(pdf_path: str) -> int:
        """PDF 총 페이지 수 반환"""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            logger.info(f"📄 PDF 페이지 수: {Path(pdf_path).name} = {total_pages}페이지")
            return total_pages
            
        except Exception as e:
            logger.error(f"❌ PDF 페이지 수 확인 오류: {e}")
            return 1
    
    @staticmethod
    def get_pdf_info(pdf_path: str) -> dict:
        """PDF 기본 정보 반환"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            info = {
                "filename": Path(pdf_path).name,
                "total_pages": len(doc),
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "file_size": Path(pdf_path).stat().st_size
            }
            
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"❌ PDF 정보 조회 오류: {e}")
            return {"filename": Path(pdf_path).name, "total_pages": 1, "error": str(e)}
    
    @staticmethod
    def create_highlighted_pdf_page(
        pdf_path: str, 
        page_num: int, 
        highlight_chunks: list
    ) -> str:
        """PDF 페이지에 하이라이트를 추가하고 임시 파일 경로 반환"""
        try:
            logger.info(f"PDF 하이라이트 생성: {Path(pdf_path).name} 페이지 {page_num + 1}")
            
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            
            # 해당 페이지의 하이라이트 정보만 필터링
            page_highlights = [
                chunk for chunk in highlight_chunks 
                if chunk.get('page_num', 0) == page_num
            ]
            
            # 하이라이트 추가
            for chunk in page_highlights:
                similarity = chunk.get('similarity_score', 0.5)
                
                # 유사도에 따른 색상 결정
                if similarity > 0.8:
                    color = (1.0, 0.0, 0.0)  # 빨간색
                elif similarity > 0.6:
                    color = (1.0, 0.5, 0.0)  # 주황색
                else:
                    color = (1.0, 1.0, 0.0)  # 노란색
                
                # 청크 텍스트로 직접 검색하여 하이라이트
                chunk_text = chunk.get('text', '')
                if chunk_text.strip():
                    # 청크 텍스트를 줄별로 분할
                    chunk_sentences = chunk_text.split('\n')
                    
                    for sentence in chunk_sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 10:  # 너무 짧은 문장은 제외
                            # 페이지에서 해당 문장 검색
                            text_instances = page.search_for(sentence[:50])  # 처음 50자로 검색
                            
                            for inst in text_instances:
                                try:
                                    # 하이라이트 추가
                                    highlight_annot = page.add_highlight_annot(inst)
                                    highlight_annot.set_colors(stroke=color)
                                    highlight_annot.update()
                                except Exception as e:
                                    logger.debug(f"하이라이트 추가 실패: {e}")
            
            # 페이지를 고해상도 이미지로 변환
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(img_data)
                temp_file_path = temp_file.name
            
            doc.close()
            logger.info(f"✅ PDF 하이라이트 생성 완료: {temp_file_path}")
            return temp_file_path
            
        except Exception as e:
            logger.error(f"❌ PDF 하이라이트 생성 오류: {e}")
            # 하이라이트 실패 시 일반 페이지 반환
            return PDFUtils.get_pdf_page_image(pdf_path, page_num)
    
    @staticmethod
    def extract_text_from_page(pdf_path: str, page_num: int) -> str:
        """특정 페이지에서 텍스트 추출"""
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            text = page.get_text()
            doc.close()
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"❌ 페이지 텍스트 추출 오류: {e}")
            return ""
    
    @staticmethod
    def search_text_in_pdf(pdf_path: str, search_term: str) -> list:
        """PDF에서 텍스트 검색"""
        try:
            doc = fitz.open(pdf_path)
            results = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_instances = page.search_for(search_term)
                
                if text_instances:
                    results.append({
                        "page_num": page_num,
                        "matches": len(text_instances),
                        "rectangles": [list(rect) for rect in text_instances]
                    })
            
            doc.close()
            return results
            
        except Exception as e:
            logger.error(f"❌ PDF 텍스트 검색 오류: {e}")
            return []

class TempFileManager:
    """임시 파일 관리"""
    
    def __init__(self):
        self.temp_files = []
    
    def add_temp_file(self, file_path: str):
        """임시 파일 목록에 추가"""
        if file_path and file_path not in self.temp_files:
            self.temp_files.append(file_path)
    
    def cleanup_temp_files(self):
        """임시 파일들 정리"""
        
        cleaned_count = 0
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {temp_file}, {e}")
        
        self.temp_files.clear()
        
        if cleaned_count > 0:
            logger.info(f"🗑️  임시 파일 정리 완료: {cleaned_count}개")

def test_pdf_utils():
    """PDF 유틸리티 테스트"""
    from config import PDF_FILES
    
    print("🧪 PDF 유틸리티 테스트")
    
    # PDF 파일이 있는 경우 테스트
    existing_pdfs = [pdf for pdf in PDF_FILES if Path(pdf).exists()]
    
    if existing_pdfs:
        test_pdf = existing_pdfs[0]
        print(f"📄 테스트 PDF: {Path(test_pdf).name}")
        
        # PDF 정보 조회
        info = PDFUtils.get_pdf_info(test_pdf)
        print(f"📊 PDF 정보: {info['total_pages']}페이지, {info['file_size']:,}바이트")
        
        # 첫 페이지 이미지 생성
        image_path = PDFUtils.get_pdf_page_image(test_pdf, 0)
        if image_path:
            print(f"🖼️  페이지 이미지 생성: {image_path}")
            
            # 임시 파일 정리
            temp_manager = TempFileManager()
            temp_manager.add_temp_file(image_path)
            temp_manager.cleanup_temp_files()
        
        # 텍스트 검색 테스트
        search_results = PDFUtils.search_text_in_pdf(test_pdf, "사업")
        print(f"🔍 텍스트 검색 결과: {len(search_results)}페이지에서 발견")
        
        return True
    else:
        print("❌ 테스트용 PDF 파일이 없습니다.")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_pdf_utils()
