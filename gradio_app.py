"""
Gradio 웹 애플리케이션
EXAONE 4.0-32B-AWQ 기반 사용자 인터페이스 및 상호작용 관리
"""

import logging
import gradio as gr
from pathlib import Path

from config import (
    PROJECT_NAME, VERSION, GRADIO_HOST, GRADIO_PORT, GRADIO_THEME,
    PDF_FILES, ensure_directories
)
from chat_manager import ChatManager
from pdf_utils import PDFUtils, TempFileManager

logger = logging.getLogger(__name__)

class GradioApp:
    """EXAONE 4.0-32B-AWQ 기반 Gradio 웹 애플리케이션 클래스"""
    
    def __init__(self):
        # 디렉토리 확인
        ensure_directories()
        
        # 채팅 매니저 초기화
        self.chat_manager = ChatManager()
        
        # PDF 유틸리티 및 임시 파일 관리
        self.pdf_utils = PDFUtils()
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
            # 파일명에서 확장자 제거하고 번호 추가
            filename = Path(pdf_path).stem
            choices.append(f"{i}. {filename}")
        
        return choices if choices else ["PDF 파일이 없습니다"]
    
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
                        type="tuples",  # 기존 코드와 호환을 위해 다시 tuples로 변경
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="메시지 입력",
                            placeholder="예: RISE 사업비 집행 시 주의사항이 무엇인가요?",
                            lines=2,
                            show_label=True,
                            scale=4
                        )
                        with gr.Row():
                            submit = gr.Button("전송", variant="primary", scale=1)
                            clear = gr.Button("대화 초기화", variant="secondary")
                    
                    with gr.Row():
                        reasoning_mode = gr.Checkbox(
                            label="🧠 Reasoning 모드",
                            value=True,
                        )
                        enable_reasoning = gr.Checkbox(
                            label="🤔 AI 추론과정 표시",
                            value=True,
                        )
                        agent_mode = gr.Checkbox(
                            label="🤖 AI Agent 모드",
                            value=True,
                        )
                        enable_rag = gr.Checkbox(
                            label="📚 문서 내 검색",
                            value=True,
                        )

                    with gr.Row():
                        pdf_selector = gr.Dropdown(
                            choices=self.chat_manager.get_pdf_list(),
                            value="all",
                            label="📚 검색 대상 문서 선택",
                            info="답변에 참고할 문서를 선택하세요"
                        )
                
                with gr.Column(scale=5):
                    gr.Markdown("### 📄 원본 문서 뷰어")
                    
                    # PDF 이미지 표시
                    pdf_image = gr.Image(
                        label="PDF 페이지",
                        height=400,
                        show_label=False
                    )
                    
                    # PDF 선택 및 네비게이션
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
                    
                    # RAG 검색 결과 표시
                    gr.Markdown("### 🔍 관련 문서 내용")
                    doc_info = gr.Markdown(
                        "질문을 입력하면 관련 문서 내용이 여기에 표시됩니다.",
                        height=300)
                    
                    # thinking_display를 숨겨진 컴포넌트로 유지 (오류 방지)
                    thinking_display = gr.Markdown(
                        "추론 과정이 활성화되면 AI의 사고 과정이 여기에 표시됩니다.",
                        visible=False)
            
            # 상태 변수들
            current_pdf_index = gr.State(0)
            current_page = gr.State(0)
            
            # 이벤트 핸들러 정의
            def user_input(user_message, history, reasoning_mode, enable_reasoning, agent_mode, enable_rag, selected_pdf):
                """사용자 입력 처리"""
                if not user_message:
                    yield (history, "", "", "", gr.Button(interactive=True))
                    return
                
                # 제출 버튼 비활성화
                submit_btn_enabled = gr.Button(interactive=False)
                
                # 초기 변수들 설정
                thinking_display = "추론 과정이 활성화되면 AI의 사고 과정이 여기에 표시됩니다."
                doc_info_display = "🔄 관련 문서를 검색하고 있습니다..."
                thinking_content = ""  # 추론 내용 초기화
                clean_doc_info = "질문을 입력하면 관련 문서 내용이 여기에 표시됩니다."  # 초기화 추가
                
                # 선택된 PDF 경로 결정
                pdf_path = None if selected_pdf == "all" else selected_pdf
                
                
                for result in self.chat_manager.process_message(
                    user_message, 
                    history or [], 
                    agent_mode=agent_mode,  # AI Agent 모드 활성화 여부
                    reasoning_mode=reasoning_mode,  # 추론 모드 활성화 여부
                    enable_reasoning=enable_reasoning,  # 추론 과정 표시 여부
                    enable_rag=enable_rag, # 문서 내 검색 활성화 여부
                    selected_pdf_path=pdf_path  # 선택된 PDF 경로
                ):
                    # result는 (new_history, doc_info) 튜플
                    new_history, raw_doc_info = result
                 
                    # enable_reasoning과 reasoning_mode 모두 활성일 때만 추론 패널 표시
                    if enable_reasoning and reasoning_mode:
                        # 최신 응답에서 추론 섹션을 추출하여 별도 패널에도 반영
                        panel_thinking = ""
                        if new_history and len(new_history) > 0:
                            last_response = new_history[-1][1] or ""
                            think_marker = "🤔 **[추론 과정]**"
                            answer_marker = "💬 **[최종 답변]**"
                            if think_marker in last_response:
                                if answer_marker in last_response:
                                    # 추론과 답변이 모두 있는 경우
                                    parts = last_response.split(answer_marker)
                                    thinking_part = parts[0].replace(think_marker, "").strip()
                                    if thinking_part:
                                        panel_thinking = thinking_part
                                else:
                                    # 추론만 있는 경우 (아직 생성 중)
                                    panel_thinking = last_response.replace(think_marker, "").strip()
                        
                        # API 호출 상태가 포함된 메시지에서 상태 정보 추출
                        status_info = ""
                        if new_history and len(new_history) > 0:
                            last_msg = new_history[-1][1] or ""
                            if "📡 API 호출 상태:" in last_msg:
                                # API 상태 정보를 추출하여 별도 표시
                                lines = last_msg.split("\\n")
                                for line in lines:
                                    if "📡 API 호출 상태:" in line:
                                        status_info = line.strip()
                                        break
                        
                        # 추론 패널 업데이트 (API 상태 포함)
                        if panel_thinking:
                            thinking_display = f"{panel_thinking}\\n\\n{status_info}" if status_info else panel_thinking
                        else:
                            thinking_display = status_info if status_info else thinking_content
                        
                        yield new_history, thinking_display, raw_doc_info, "", submit_btn_enabled
                    else:
                        # 비표시: 추론 패널 비움
                        yield new_history, "", raw_doc_info, "", submit_btn_enabled
                
                # 처리 완료 후 제출 버튼 재활성화
                submit_btn_reactivated = gr.Button(interactive=True)
                
                # 최종 상태 확인 및 보정
                if 'new_history' in locals() and new_history:
                    # 최종 응답이 비어있거나 불완전한 경우 확인
                    if len(new_history) > 0 and new_history[-1][1]:
                        final_response = new_history[-1][1]
                        if final_response.strip() and not final_response.endswith("🔄"):
                            # 정상적인 최종 응답이 있는 경우
                            yield new_history, thinking_display if (enable_reasoning and reasoning_mode) else "", raw_doc_info, "", submit_btn_reactivated
                        else:
                            # 불완전한 응답인 경우 메시지 추가
                            new_history[-1] = [new_history[-1][0], final_response + "\n\n⚠️ 응답이 완료되었습니다."]
                            yield new_history, thinking_display if (enable_reasoning and reasoning_mode) else "", raw_doc_info, "", submit_btn_reactivated
                    else:
                        # 응답이 없는 경우 오류 메시지 표시
                        error_history = history + [[user_message, "❌ 응답 생성 중 오류가 발생했습니다. 다시 시도해주세요."]]
                        yield error_history, "", "질문을 입력하면 관련 문서 내용이 여기에 표시됩니다.", "", submit_btn_reactivated
                else:
                    # new_history가 없는 경우
                    error_history = history + [[user_message, "❌ 응답을 생성할 수 없습니다. 다시 시도해주세요."]]
                    yield error_history, "", "질문을 입력하면 관련 문서 내용이 여기에 표시됩니다.", "", submit_btn_reactivated
            
            def clear_history():
                """대화 기록 초기화"""
                self.temp_file_manager.cleanup_temp_files()
                return [], "질문을 입력하면 관련 문서 내용이 여기에 표시됩니다.", "추론 과정이 활성화되면 AI의 사고 과정이 여기에 표시됩니다."
            
            def change_pdf(pdf_choice, current_pdf_idx, current_pg):
                """PDF 변경"""
                # PDF 선택에서 번호 추출 (예: "1. 파일명" -> 0)
                try:
                    new_pdf_idx = int(pdf_choice.split(".")[0]) - 1
                    if new_pdf_idx < 0 or new_pdf_idx >= len(self.existing_pdfs):
                        new_pdf_idx = 0
                except (ValueError, IndexError):
                    new_pdf_idx = 0
                
                image, page_info_text = self.get_pdf_page(new_pdf_idx, 0)
                return image, page_info_text, new_pdf_idx, 0
            
            def prev_page(pdf_idx, page):
                """이전 페이지"""
                if page > 0:
                    new_page = page - 1
                    image, page_info_text = self.get_pdf_page(pdf_idx, new_page)
                    return image, page_info_text, new_page
                return None, f"Page {page + 1} / ?", page
            
            def next_page(pdf_idx, page):
                """다음 페이지"""
                if pdf_idx < len(self.existing_pdfs):
                    total_pages = self.pdf_utils.get_pdf_total_pages(self.existing_pdfs[pdf_idx])
                    if page < total_pages - 1:
                        new_page = page + 1
                        image, page_info_text = self.get_pdf_page(pdf_idx, new_page)
                        return image, page_info_text, new_page
                return None, f"Page {page + 1} / ?", page
            
            def load_initial_pdf():
                """초기 PDF 로드"""
                if self.existing_pdfs and Path(self.existing_pdfs[0]).exists():
                    image, page_info_text = self.get_pdf_page(0, 0)
                    return image, page_info_text
                return None, "PDF 파일이 없습니다."
            
            # 이벤트 연결
            submit.click(
                user_input,
                inputs=[msg, chatbot, reasoning_mode, enable_reasoning, agent_mode, enable_rag, pdf_selector],
                outputs=[chatbot, thinking_display, doc_info, msg, submit],  # 5개 outputs로 수정
                api_name=False
            )
            
            clear.click(
                clear_history,
                outputs=[chatbot, doc_info, thinking_display],
                api_name=False
            )
            
            msg.submit(
                user_input,
                inputs=[msg, chatbot, reasoning_mode, enable_reasoning, agent_mode, enable_rag, pdf_selector],
                outputs=[chatbot, thinking_display, doc_info, msg, submit],  # 5개 outputs로 수정
                api_name=False
            )
            
            # PDF 네비게이션 이벤트
            pdf_viewer_selector.change(
                change_pdf,
                inputs=[pdf_viewer_selector, current_pdf_index, current_page],
                outputs=[pdf_image, page_info, current_pdf_index, current_page]
            )
            
            prev_btn.click(
                prev_page,
                inputs=[current_pdf_index, current_page],
                outputs=[pdf_image, page_info, current_page]
            )
            
            next_btn.click(
                next_page,
                inputs=[current_pdf_index, current_page],
                outputs=[pdf_image, page_info, current_page]
            )
            
            # 앱 로드 시 초기화
            demo.load(
                load_initial_pdf,
                outputs=[pdf_image, page_info]
            )
        
        # 스트리밍 성능 향상을 위한 설정
        demo.queue(default_concurrency_limit=1, max_size=10)
        
        return demo
    
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
            
            # PDF 페이지 이미지 생성
            image_path = self.pdf_utils.get_pdf_page_image(pdf_path, page_num)
            
            # 임시 파일 관리에 추가
            if image_path:
                self.temp_file_manager.add_temp_file(image_path)
            
            return image_path, f"Page {page_num + 1} of {total_pages}"
            
        except Exception as e:
            logger.error(f"PDF 페이지 로드 오류: {e}")
            return None, f"PDF 로드 실패: {str(e)}"
    
    def launch(self, host: str = GRADIO_HOST, port: int = GRADIO_PORT, share: bool = False):
        """애플리케이션 실행"""
        demo = self.create_interface()
        
        logger.info(f"🚀 EXAONE Gradio 앱 실행: {host}:{port}")
        
        if share:
            demo.launch(
                server_name=host,
                server_port=port,
                share=True
            )
        else:
            demo.launch(
                server_name=host,
                server_port=port,
                share=False
            )

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
    print("  - LLM: LGAI-EXAONE/EXAONE-4.0-32B-GPTQ (vLLM 서버)")
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
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 시작 정보 출력
    print_startup_info()
    print(f"📍 접속 URL: http://localhost:{args.port}")
    print(f"🤖 vLLM 서버: http://100.105.243.84:8010/v1")
    print(f"📁 데이터 디렉토리: {Path().absolute()}")
    print(f"{'='*80}")
    
    try:
        # 애플리케이션 실행
        app = GradioApp()
        app.launch(host=args.host, port=args.port, share=args.share)
        
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"❌ 애플리케이션 실행 실패: {e}")
        print("\n💡 문제 해결 방법:")
        print("  1. vLLM 서버가 실행 중인지 확인하세요:")
        print("     vllm serve LGAI-EXAONE/EXAONE-4.0-32B-GPTQ \\")
        print("         --enable-reasoning --reasoning-parser exaone")
        print("  2. 필요한 패키지가 설치되어 있는지 확인하세요:")
        print("     pip install -r requirements.txt")
        print("  3. PDF 파일이 data/ 디렉토리에 있는지 확인하세요.")
    finally:
        # 정리 작업
        try:
            app.temp_file_manager.cleanup_temp_files()
            logger.info("🗑️  정리 작업 완료")
        except:
            pass 