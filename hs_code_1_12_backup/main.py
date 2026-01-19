"""
[Main Entry Point]
Gradio 웹 애플리케이션: 사용자의 입력을 받아 적절한 모듈(OCR, Chat, Report)로 라우팅하는 관제탑 역할
"""

import logging
import gradio as gr
from pathlib import Path

# 설정 및 내부 모듈 임포트
from config import (
    PROJECT_NAME, VERSION, GRADIO_HOST, GRADIO_PORT, GRADIO_THEME,
    PDF_FILES, ensure_directories, VLLM_SERVER_URL
)
from core.chat_manager import ChatManager
from utils.pdf_handler import PDFUtils, TempFileManager
from services.ocr_service import OCRProcessor
from core.generator import ReportGenerator

logger = logging.getLogger(__name__)

class GradioApp:
    """EXAONE 4.0-32B-AWQ 기반 Gradio 웹 애플리케이션 클래스"""
    
    def __init__(self):
        # 1. 초기화: 시스템 구동에 필요한 핵심 모듈들을 로드합니다.
        ensure_directories()
        
        # [Brain] 대화 관리자: 일반 대화, RAG 검색, 에이전트 실행을 총괄
        self.chat_manager = ChatManager()
        
        # [Planner] 보고서 생성기: 문서 구조 분석 -> 툴 플래닝 -> 보고서 작성 담당
        self.report_generator = ReportGenerator()
        
        # [Eyes] PDF/OCR 처리기: PDF 파일을 이미지나 텍스트로 변환
        self.pdf_utils = PDFUtils()
        self.ocr_processor = OCRProcessor()
        self.temp_file_manager = TempFileManager()
        
        # PDF 파일 목록 준비
        self.existing_pdfs = [pdf for pdf in PDF_FILES if Path(pdf).exists()]
        
        # 시스템 초기화
        if self.existing_pdfs:
            logger.info(f"시스템 초기화: {len(self.existing_pdfs)}개 PDF 파일")
            self.chat_manager.initialize_system(self.existing_pdfs)
        else:
            logger.warning("PDF 파일이 없습니다. data/ 디렉토리에 PDF 파일을 추가하세요.")
        
        logger.info("EXAONE Gradio 앱 초기화 완료")
    
    def get_pdf_choices(self):
        """PDF 파일 목록을 선택 옵션으로 변환"""
        choices = []
        for i, pdf_path in enumerate(self.existing_pdfs, 1):
            filename = Path(pdf_path).stem
            choices.append(f"{i}. {filename}")
        
        return choices if choices else ["PDF 파일이 없습니다"]
    
    def get_pdf_page(self, pdf_index: int, page_num: int):
        """PDF 페이지 이미지 반환"""
        try:
            if pdf_index >= len(self.existing_pdfs) or page_num < 0:
                return None, "잘못된 페이지 번호입니다."
            
            pdf_path = self.existing_pdfs[pdf_index]
            
            if not Path(pdf_path).exists():
                return None, "PDF 파일이 없습니다."
            
            total_pages = self.pdf_utils.get_pdf_total_pages(pdf_path)
            
            if page_num >= total_pages:
                return None, f"페이지 번호가 범위를 벗어났습니다. (최대: {total_pages})"
            
            image_path = self.pdf_utils.get_pdf_page_image(pdf_path, page_num)
            
            if image_path:
                self.temp_file_manager.add_temp_file(image_path)
            
            return image_path, f"Page {page_num + 1} of {total_pages}"
            
        except Exception as e:
            logger.error(f"PDF 페이지 로드 오류: {e}")
            return None, f"PDF 로드 실패: {str(e)}"

    def create_interface(self):
        """Gradio 인터페이스 생성"""
        with gr.Blocks(title=f"{PROJECT_NAME} {VERSION}", theme=GRADIO_THEME) as demo:
            gr.Markdown(f"# 🏢 {PROJECT_NAME}")

            with gr.Row():
                # 좌측: 채팅 영역
                with gr.Column(scale=5):
                    chatbot = gr.Chatbot(
                        label="💬 대화 (EXAONE 4.0 Reasoning 모드)",
                        height=500,
                        show_label=True,
                        type="tuples",  
                    )
                    
                    with gr.Row():
                        upload_btn = gr.UploadButton("📁 파일 업로드 (PDF/TXT)", file_types=[".pdf", ".txt"])
                        
                        msg = gr.Textbox(
                            label="메시지 입력",
                            placeholder="예: 이 파일을 분석해서 일일보고서를 생성해줘.",
                            lines=2,
                            show_label=True,
                            scale=4
                        )
                        with gr.Row():
                            submit = gr.Button("전송", variant="primary", scale=1)
                            clear = gr.Button("대화 초기화", variant="secondary")
                    
                    # 추천 프롬프트
                    gr.Examples(
                        examples=[
                            ["📝 [파일 업로드 후] 이 문서를 바탕으로 일일보고서 생성해줘."],
                            ["🌤️ 오늘 서울 날씨와 미세먼지 알려줘."],
                            ["⚠️ 현재 발효 중인 기상 특보가 있어?"],
                            ["📚 한파 주의보 발령 시 조치 사항은 뭐야? (매뉴얼 검색)"],
                            ["🔍 최근 3일간 발생한 지진 정보 알려줘."]
                        ],
                        inputs=msg,
                        label="💡 추천 질문 (클릭해서 입력)"
                    )

                    with gr.Group():
                        report_download = gr.File(label="다운로드", visible=False, interactive=False)
                        fact_view = gr.Markdown(label="수집된 데이터 확인", visible=False)

                    with gr.Row():
                        reasoning_mode = gr.Checkbox(label="🧠 Reasoning 모드", value=True, visible=False)
                        enable_reasoning = gr.Checkbox(label="🤔 추론과정 표시", value=True, visible=True)
                        agent_mode = gr.Checkbox(label="🤖 Agent 모드", value=True, visible=False)
                        enable_rag = gr.Checkbox(label="📚 문서 검색", value=True, visible=False)

                    with gr.Row():
                        pdf_selector = gr.Dropdown(
                            choices=self.chat_manager.get_pdf_list(),
                            value="all",
                            label="📚 검색 대상 문서 선택",
                            info="답변에 참고할 문서를 선택하세요"
                        )
                
                with gr.Column(scale=5):
                    gr.Markdown("### 📄 원본 문서 뷰어")
                    
                    pdf_image = gr.Image(label="PDF 페이지", height=400, show_label=False)
                    
                    with gr.Row():
                        pdf_choices = self.get_pdf_choices()
                        pdf_viewer_selector = gr.Dropdown(
                            choices=pdf_choices,
                            value=pdf_choices[0] if pdf_choices else None,
                            label="PDF 선택",
                            scale=3
                        )
                    
                    with gr.Row():
                        prev_btn = gr.Button("◀ 이전", scale=1)
                        page_info = gr.Textbox(
                            value="Page 1 / 1",
                            label="페이지 정보",
                            interactive=False,
                            scale=2
                        )
                        next_btn = gr.Button("다음 ▶", scale=1)
                    
                    gr.Markdown("### 🔍 관련 문서 내용")
                    doc_info = gr.Markdown("질문을 입력하면 관련 문서 내용이 여기에 표시됩니다.", height=300)
                    
                    thinking_display = gr.Markdown("추론 과정이 활성화되면 AI의 사고 과정이 여기에 표시됩니다.", visible=False)
            
            # 상태 변수들
            current_pdf_index = gr.State(0)
            current_page = gr.State(0)
            uploaded_file_state = gr.State(None)
            
            # --- 이벤트 핸들러 정의 (함수 내부에 정의하여 클로저로 활용) ---
            def user_input(user_message, history, reasoning_mode, enable_reasoning, agent_mode, enable_rag, selected_pdf, uploaded_file):
                
                # 1. [입력 단계] 파일 업로드 처리
                # 사용자가 PDF를 업로드했다면 여기서 OCR 프로세스가 시작됩니다.
                uploaded_context = ""
                file_name_display = ""
                
                if uploaded_file:
                    try:
                        file_path = uploaded_file.name
                        file_name_display = Path(file_path).name
                        
                        # [Flow 1] PDF 입력 -> OCR 변환
                        if file_path.lower().endswith(".pdf"):
                            logger.info(f"📂 PDF 파일 감지 (OCR 처리): {file_path}")
                            # OCRProcessor가 Mistral API 또는 PyMuPDF를 사용해 텍스트 추출
                            uploaded_context = self.ocr_processor.extract_pdf_text(file_path)
                        else:
                            # 텍스트 파일은 그대로 읽음
                            with open(file_path, "r", encoding="utf-8") as f:
                                uploaded_context = f.read()
                        
                        if not user_message:
                            user_message = "이 파일을 분석해서 보고서를 작성해줘."
                        
                    except Exception as e:
                        logger.error(f"파일 읽기 실패: {e}")
                        uploaded_context = f"(파일 읽기 실패: {str(e)})"
                
                # 빈 메시지 처리
                if not user_message:
                    yield (history, "", "", "", gr.Button(interactive=True), None)
                    return
                
                submit_btn_enabled = gr.Button(interactive=False)
                
                thinking_display = "추론 과정이 활성화되면 AI의 사고 과정이 여기에 표시됩니다."
                thinking_content = ""
                pdf_path = None if selected_pdf == "all" else selected_pdf
                
                # 2. [처리 단계] ChatManager에게 작업 위임
                # 추출된 텍스트(uploaded_context)와 사용자 질문(user_message)을 넘깁니다.
                # ChatManager 내부에서 '보고서 생성'인지 '일반 대화'인지 판단하여 분기합니다.
                for result in self.chat_manager.process_message(
                    user_message, 
                    history or [], 
                    agent_mode, 
                    reasoning_mode, 
                    enable_reasoning, 
                    enable_rag, 
                    selected_pdf_path=pdf_path,
                    uploaded_context=uploaded_context # [중요] 방금 업로드해서 OCR한 내용
                ):
                    # 3. [출력 단계] 실시간 스트리밍 응답 처리
                    new_history, raw_doc_info = result
                    
                    # [UI Feedback] 파일 업로드 시 응답 상단에 파일명 표시
                    if file_name_display and new_history and len(new_history) > 0:
                        upload_msg = f"📂 **파일 업로드됨:** `{file_name_display}` (분석 중...)\n\n"
                        if not new_history[-1][1].startswith("📂 **파일 업로드됨"):
                            new_history[-1][1] = upload_msg + new_history[-1][1]
                    
                    if enable_reasoning and reasoning_mode:
                        panel_thinking = ""
                        if new_history and len(new_history) > 0:
                            last_response = new_history[-1][1] or ""
                            think_marker = "🤔 **[추론 과정]**"
                            if think_marker in last_response:
                                parts = last_response.split("💬 **[최종 답변]**")
                                panel_thinking = parts[0].replace(think_marker, "").strip()
                                # 파일 업로드 메시지가 추론 패널에 들어가지 않도록 제거
                                if file_name_display:
                                    panel_thinking = panel_thinking.replace(f"📂 **파일 업로드됨:** `{file_name_display}` (분석 중...)\n\n", "")
                        
                        status_info = ""
                        if new_history and len(new_history) > 0:
                            if "📡 API 호출 상태:" in new_history[-1][1]:
                                status_info = "📡 API 호출 중..."

                        if panel_thinking:
                            thinking_display = f"{panel_thinking}\n\n{status_info}" if status_info else panel_thinking
                        else:
                            thinking_display = status_info if status_info else thinking_content
                        
                        yield new_history, thinking_display, raw_doc_info, "", submit_btn_enabled, None
                    else:
                        yield new_history, "", raw_doc_info, "", submit_btn_enabled, None
                
                submit_btn_reactivated = gr.Button(interactive=True)
                
                if 'new_history' in locals() and new_history:
                    if len(new_history) > 0 and new_history[-1][1]:
                        final_response = new_history[-1][1]
                        if final_response.strip() and not final_response.endswith("🔄"):
                            yield new_history, thinking_display if (enable_reasoning and reasoning_mode) else "", raw_doc_info, "", submit_btn_reactivated, None
                        else:
                            new_history[-1] = [new_history[-1][0], final_response + "\n\n⚠️ 응답이 완료되었습니다."]
                            yield new_history, thinking_display if (enable_reasoning and reasoning_mode) else "", raw_doc_info, "", submit_btn_reactivated, None
                    else:
                        error_history = history + [[user_message, "❌ 응답 생성 중 오류가 발생했습니다."]]
                        yield error_history, "", "오류 발생", "", submit_btn_reactivated, None
                else:
                    error_history = history + [[user_message, "❌ 응답을 생성할 수 없습니다."]]
                    yield error_history, "", "오류 발생", "", submit_btn_reactivated, None

            def clear_history():
                self.temp_file_manager.cleanup_temp_files()
                return [], "질문을 입력하면 관련 문서 내용이 여기에 표시됩니다.", "추론 과정이 활성화되면 AI의 사고 과정이 여기에 표시됩니다.", gr.File(visible=False), None
            
            def handle_upload(file):
                return file

            # --- PDF Viewer Handlers ---
            def change_pdf(pdf_choice, current_pdf_idx, current_pg):
                try:
                    new_pdf_idx = int(pdf_choice.split(".")[0]) - 1
                    if new_pdf_idx < 0 or new_pdf_idx >= len(self.existing_pdfs): new_pdf_idx = 0
                except: new_pdf_idx = 0
                
                image, page_info_text = self.get_pdf_page(new_pdf_idx, 0)
                return image, page_info_text, new_pdf_idx, 0
            
            def prev_page(pdf_idx, page):
                if page > 0:
                    new_page = page - 1
                    image, page_info_text = self.get_pdf_page(pdf_idx, new_page)
                    return image, page_info_text, new_page
                return None, f"Page {page + 1} / ?", page
            
            def next_page(pdf_idx, page):
                if pdf_idx < len(self.existing_pdfs):
                    total_pages = self.pdf_utils.get_pdf_total_pages(self.existing_pdfs[pdf_idx])
                    if page < total_pages - 1:
                        new_page = page + 1
                        image, page_info_text = self.get_pdf_page(pdf_idx, new_page)
                        return image, page_info_text, new_page
                return None, f"Page {page + 1} / ?", page
            
            def load_initial_pdf():
                if self.existing_pdfs and Path(self.existing_pdfs[0]).exists():
                    image, page_info_text = self.get_pdf_page(0, 0)
                    return image, page_info_text
                return None, "PDF 파일이 없습니다."

            # --- Event Binding ---
            upload_btn.upload(handle_upload, inputs=[upload_btn], outputs=[uploaded_file_state])

            submit.click(
                user_input,
                inputs=[msg, chatbot, reasoning_mode, enable_reasoning, agent_mode, enable_rag, pdf_selector, uploaded_file_state],
                outputs=[chatbot, thinking_display, doc_info, msg, submit, uploaded_file_state],
                api_name=False
            )
            
            clear.click(
                clear_history,
                outputs=[chatbot, doc_info, thinking_display, report_download, uploaded_file_state],
                api_name=False
            )
            
            msg.submit(
                user_input,
                inputs=[msg, chatbot, reasoning_mode, enable_reasoning, agent_mode, enable_rag, pdf_selector, uploaded_file_state],
                outputs=[chatbot, thinking_display, doc_info, msg, submit, uploaded_file_state],
                api_name=False
            )
            
            pdf_viewer_selector.change(
                change_pdf,
                inputs=[pdf_viewer_selector, current_pdf_index, current_page],
                outputs=[pdf_image, page_info, current_pdf_index, current_page]
            )
            
            prev_btn.click(prev_page, inputs=[current_pdf_index, current_page], outputs=[pdf_image, page_info, current_page])
            next_btn.click(next_page, inputs=[current_pdf_index, current_page], outputs=[pdf_image, page_info, current_page])
            
            demo.load(load_initial_pdf, outputs=[pdf_image, page_info])
        
        demo.queue(default_concurrency_limit=1, max_size=10)
        return demo
    
    def launch(self, host: str = GRADIO_HOST, port: int = GRADIO_PORT, share: bool = False):
        demo = self.create_interface()
        logger.info(f"🚀 EXAONE Gradio 앱 실행: {host}:{port}")
        if share:
            demo.launch(server_name=host, server_port=port, share=True)
        else:
            demo.launch(server_name=host, server_port=port, share=False)

def print_startup_info():
    """시작 정보 출력"""
    print(f"\n{'='*80}")
    print(f"🏢 {PROJECT_NAME} {VERSION}")
    print(f"{'='*80}")
    print("✅ 주요 기능:")
    print("  1. ✅ EXAONE 4.0-32B-AWQ Reasoning 모델")
    print("  2. ✅ vLLM 서버 기반 고속 추론")
    print("  3. ✅ <think> 추론과정 표시")
    print("  4. ✅ LangChain 기반 RAG 시스템")
    print("  5. ✅ FAISS 벡터 데이터베이스")
    print("  6. ✅ Mistral OCR + 자동 캐싱")
    print("  7. ✅ 실시간 스트리밍 응답")
    print("  8. ✅ PDF 뷰어 통합")
    print("  9. ✅ 모듈화된 아키텍처")
    print(f"{'='*80}")
    print("🔧 기술 스택:")
    print("  - LLM: LGAI-EXAONE/EXAONE-4.0-32B-AWQ (vLLM 서버)")
    print("  - Reasoning: <think> 추론 모드")
    print("  - Embedding: dragonkue/BGE-m3-ko")
    print("  - Vector DB: FAISS (GPU 가속)")
    print("  - OCR: Mistral OCR API")
    print("  - Framework: LangChain + Gradio")
    print(f"{'='*80}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} {VERSION}')
    parser.add_argument('--host', type=str, default=GRADIO_HOST, help='Host IP address')
    parser.add_argument('--port', type=int, default=GRADIO_PORT, help='Port number')
    parser.add_argument('--share', action='store_true', default=True, help='Create a public URL (default: True)')
    parser.add_argument('--no-share', dest='share', action='store_false', help='Disable public URL')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    print_startup_info()
    print(f"📍 접속 URL: http://localhost:{args.port}")
    print(f"🤖 vLLM 서버: {VLLM_SERVER_URL}")
    
    try:
        app = GradioApp()
        app.launch(host=args.host, port=args.port, share=args.share)
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"❌ 애플리케이션 실행 실패: {e}")
        print("\n💡 문제 해결 방법:")
        print("  1. vLLM 서버가 실행 중인지 확인하세요.")
        print("  2. 필요한 패키지가 설치되어 있는지 확인하세요.")
        print("  3. PDF 파일이 data/ 디렉토리에 있는지 확인하세요.")
    finally:
        try:
            app.temp_file_manager.cleanup_temp_files()
            logger.info("🗑️  정리 작업 완료")
        except: pass