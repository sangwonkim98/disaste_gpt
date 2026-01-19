"""
[LangGraph 기반 Main Entry Point]
Gradio 웹 애플리케이션: LangGraph 그래프를 사용한 새로운 진입점

기존 main.py의 UI 구조를 유지하면서 ChatManager.process_message() 대신
LangGraph의 graph.invoke() / graph.stream()을 사용합니다.
"""

import logging
import uuid
import gradio as gr
from pathlib import Path
from typing import Generator, Tuple, List, Dict, Any

# 설정 및 내부 모듈 임포트
from config import (
    PROJECT_NAME, VERSION, GRADIO_HOST, GRADIO_PORT, GRADIO_THEME,
    PDF_FILES, ensure_directories, VLLM_SERVER_URL
)
from graph import build_graph, GraphState
from graph.nodes import _get_rag_system
from utils.pdf_handler import PDFUtils, TempFileManager
from services.ocr_service import OCRProcessor
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class LangGraphGradioApp:
    """LangGraph 기반 Gradio 웹 애플리케이션 클래스"""

    def __init__(self):
        # 1. 초기화: 시스템 구동에 필요한 핵심 모듈들을 로드합니다.
        ensure_directories()

        # [LangGraph] 그래프 빌드
        logger.info("🔨 LangGraph 그래프 초기화 중...")
        self.graph = build_graph()

        # [Eyes] PDF/OCR 처리기
        self.pdf_utils = PDFUtils()
        self.ocr_processor = OCRProcessor()
        self.temp_file_manager = TempFileManager()

        # PDF 파일 목록 준비
        self.existing_pdfs = [pdf for pdf in PDF_FILES if Path(pdf).exists()]

        # RAG 시스템 초기화 (벡터스토어 빌드)
        if self.existing_pdfs:
            logger.info(f"📚 RAG 시스템 초기화: {len(self.existing_pdfs)}개 PDF 파일")
            rag_system = _get_rag_system()
            rag_system.build_index(self.existing_pdfs)
        else:
            logger.warning("PDF 파일이 없습니다. data/ 디렉토리에 PDF 파일을 추가하세요.")

        logger.info("✅ LangGraph Gradio 앱 초기화 완료")

    def get_pdf_choices(self) -> List[str]:
        """PDF 파일 목록을 선택 옵션으로 변환"""
        choices = []
        for i, pdf_path in enumerate(self.existing_pdfs, 1):
            filename = Path(pdf_path).stem
            choices.append(f"{i}. {filename}")

        return choices if choices else ["PDF 파일이 없습니다"]

    def get_pdf_list_for_dropdown(self) -> List[Tuple[str, str]]:
        """PDF 파일 목록을 드롭다운용으로 변환"""
        choices = [("전체 문서에서 검색 (All Documents)", "all")]
        for pdf_path in self.existing_pdfs:
            pdf_name = Path(pdf_path).stem
            choices.append((pdf_name, pdf_path))
        return choices

    def get_pdf_page(self, pdf_index: int, page_num: int) -> Tuple[Any, str]:
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

    def process_with_graph(
        self,
        user_message: str,
        history: List[Dict[str, str]],
        reasoning_mode: bool,
        agent_mode: bool,
        enable_rag: bool,
        selected_pdf: str,
        uploaded_context: str = ""
    ) -> Generator[Tuple[List[Dict[str, str]], str, str], None, None]:
        """
        LangGraph를 사용하여 메시지 처리 (Messages 포맷 지원)

        Args:
            user_message: 사용자 메시지
            history: 대화 히스토리 (Gradio chatbot messages 형식 [{'role': 'user', 'content': ...}, ...])
            reasoning_mode: 추론 모드 활성화 여부
            agent_mode: Agent 모드 활성화 여부
            enable_rag: RAG 활성화 여부
            selected_pdf: 선택된 PDF 경로
            uploaded_context: 업로드된 파일 내용

        Yields:
            (updated_history, doc_info, thinking_content) 튜플
        """
        # 히스토리를 LangGraph 메시지 형식으로 변환
        messages = []
        if history:
            for item in history:
                role = item.get('role')
                content = item.get('content')
                if role == 'user':
                    messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    messages.append(AIMessage(content=content))

        # 초기 상태 구성
        initial_state: GraphState = {
            "messages": messages,
            "user_input": user_message,
            "uploaded_context": uploaded_context,
            "agent_mode": agent_mode,
            "reasoning_mode": reasoning_mode,
            "enable_rag": enable_rag,
            "selected_pdf": None if selected_pdf == "all" else selected_pdf,
            "next_action": "",
            "tool_results": None,
            "rag_results": None,
            "report_output": None,
            "tool_calls": None,
            "thinking_content": None,
            "final_response": "",
            "reference_docs": None,
        }

        # 진행 중 표시 (Messages 형식)
        # 사용자 메시지 추가
        history_with_input = (history or []) + [{"role": "user", "content": user_message}]
        # 응답 플레이스홀더 추가
        new_history = history_with_input + [{"role": "assistant", "content": "🔄 처리 중..."}]
        yield new_history, "", ""

        try:
            # LangGraph 실행 (스트리밍)
            logger.info(f"🚀 LangGraph 실행 시작: {user_message[:50]}...")

            # graph.stream()을 사용하여 각 노드 실행 결과를 받음
            doc_info = ""
            thinking_content = ""
            final_response = ""

            for event in self.graph.stream(initial_state):
                # event는 {노드이름: 노드출력} 형식
                for node_name, node_output in event.items():
                    logger.info(f"📍 노드 실행: {node_name}")

                    # node_output이 None인 경우 스킵
                    if not node_output:
                        continue

                    # 참조 문서 정보 업데이트
                    if node_output.get("reference_docs"):
                        doc_info = node_output["reference_docs"]

                    # 추론 과정 업데이트
                    if node_output.get("thinking_content"):
                        thinking_content = node_output["thinking_content"]

                    # 최종 응답 업데이트
                    if node_output.get("final_response"):
                        final_response = node_output["final_response"]

                        # 히스토리 업데이트 (Messages 형식)
                        new_history[-1]['content'] = final_response
                        yield new_history, doc_info, thinking_content

            # 최종 결과 반환
            if not final_response:
                final_response = "응답을 생성하지 못했습니다."
                new_history[-1]['content'] = final_response

            yield new_history, doc_info, thinking_content

        except Exception as e:
            logger.error(f"❌ LangGraph 실행 오류: {e}")
            import traceback
            traceback.print_exc()
            error_response = f"오류가 발생했습니다: {str(e)}"
            new_history[-1]['content'] = error_response
            yield new_history, "", ""

    def create_interface(self):
        """Gradio 인터페이스 생성"""
        with gr.Blocks(title=f"{PROJECT_NAME} {VERSION} (LangGraph)") as demo:
            gr.Markdown(f"# 🏢 {PROJECT_NAME} (LangGraph Edition)")

            with gr.Row():
                # 좌측: 채팅 영역
                with gr.Column(scale=5):
                    chatbot = gr.Chatbot(
                        label="💬 대화 (LangGraph + EXAONE 4.0)",
                        height=500,
                        show_label=True,
                    )

                    with gr.Row():
                        upload_btn = gr.UploadButton("📁 파일 업로드 (PDF/TXT)", file_types=[ ".pdf", ".txt"])

                        msg = gr.Textbox(
                            label="메시지 입력",
                            placeholder="예: 오늘 서울 날씨 어때? / 한파 주의보 행동요령 알려줘",
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
                            ["🌤️ 오늘 서울 날씨와 미세먼지 알려줘."],
                            ["⚠️ 현재 발효 중인 기상 특보가 있어?"],
                            ["📚 한파 주의보 발령 시 조치 사항은 뭐야?"],
                            ["🔍 최근 3일간 발생한 지진 정보 알려줘."],
                            ["📝 [파일 업로드 후] 이 문서를 바탕으로 보고서 생성해줘."],
                        ],
                        inputs=msg,
                        label="💡 추천 질문 (클릭해서 입력)"
                    )

                    with gr.Row():
                        reasoning_mode = gr.Checkbox(label="🧠 Reasoning 모드", value=True)
                        agent_mode = gr.Checkbox(label="🤖 Agent 모드 (도구 사용)", value=True)
                        enable_rag = gr.Checkbox(label="📚 문서 검색 (RAG)", value=True)

                    with gr.Row():
                        pdf_selector = gr.Dropdown(
                            choices=[c[0] for c in self.get_pdf_list_for_dropdown()],
                            value="전체 문서에서 검색 (All Documents)",
                            label="📚 검색 대상 문서 선택",
                            info="답변에 참고할 문서를 선택하세요"
                        )

                # 우측: 문서 뷰어 및 정보 표시
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
                    doc_info = gr.Markdown("질문을 입력하면 관련 문서 내용이 여기에 표시됩니다.", height=200)

                    gr.Markdown("### 🤔 추론 과정")
                    thinking_display = gr.Markdown("Reasoning 모드가 활성화되면 AI의 사고 과정이 여기에 표시됩니다.", height=100)

            # 상태 변수들
            current_pdf_index = gr.State(0)
            current_page = gr.State(0)
            uploaded_file_state = gr.State(None)
            uploaded_context_state = gr.State("")

            # --- 이벤트 핸들러 정의 ---
            def user_input(user_message, history, reasoning_mode, agent_mode, enable_rag, selected_pdf, uploaded_context):
                """메시지 처리 핸들러"""
                if not user_message:
                    yield history, "", "", "", gr.Button(interactive=True), ""
                    return

                # PDF 선택값에서 실제 경로 추출
                pdf_list = self.get_pdf_list_for_dropdown()
                selected_pdf_path = "all"
                for name, path in pdf_list:
                    if name == selected_pdf:
                        selected_pdf_path = path
                        break

                # LangGraph로 처리
                for new_history, doc_info_content, thinking_content in self.process_with_graph(
                    user_message,
                    history or [],
                    reasoning_mode,
                    agent_mode,
                    enable_rag,
                    selected_pdf_path,
                    uploaded_context
                ):
                    yield new_history, thinking_content, doc_info_content, "", gr.Button(interactive=False), ""

                yield new_history, thinking_content, doc_info_content, "", gr.Button(interactive=True), ""

            def handle_upload(file):
                """파일 업로드 처리"""
                if file is None:
                    return None, ""

                try:
                    file_path = file.name
                    if file_path.lower().endswith(".pdf"):
                        logger.info(f"📂 PDF 파일 감지 (OCR 처리): {file_path}")
                        context = self.ocr_processor.extract_pdf_text(file_path)
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            context = f.read()

                    return file, context
                except Exception as e:
                    logger.error(f"파일 읽기 실패: {e}")
                    return file, f"(파일 읽기 실패: {str(e)})"

            def clear_history():
                """대화 초기화"""
                self.temp_file_manager.cleanup_temp_files()
                return [], "", "", None, ""

            # --- PDF Viewer Handlers ---
            def change_pdf(pdf_choice, current_pdf_idx, current_pg):
                try:
                    new_pdf_idx = int(pdf_choice.split(".")[0]) - 1
                    if new_pdf_idx < 0 or new_pdf_idx >= len(self.existing_pdfs):
                        new_pdf_idx = 0
                except:
                    new_pdf_idx = 0

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
            upload_btn.upload(
                handle_upload,
                inputs=[upload_btn],
                outputs=[uploaded_file_state, uploaded_context_state]
            )

            submit.click(
                user_input,
                inputs=[msg, chatbot, reasoning_mode, agent_mode, enable_rag, pdf_selector, uploaded_context_state],
                outputs=[chatbot, thinking_display, doc_info, msg, submit, uploaded_context_state],
                api_name=False
            )

            clear.click(
                clear_history,
                outputs=[chatbot, doc_info, thinking_display, uploaded_file_state, uploaded_context_state],
                api_name=False
            )

            msg.submit(
                user_input,
                inputs=[msg, chatbot, reasoning_mode, agent_mode, enable_rag, pdf_selector, uploaded_context_state],
                outputs=[chatbot, thinking_display, doc_info, msg, submit, uploaded_context_state],
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
        """애플리케이션 실행"""
        demo = self.create_interface()
        logger.info(f"🚀 LangGraph Gradio 앱 실행: {host}:{port}")
        demo.launch(server_name=host, server_port=port, share=share)


def print_startup_info():
    """시작 정보 출력"""
    print(f"\n{'='*80}")
    print(f"🏢 {PROJECT_NAME} {VERSION} (LangGraph Edition)")
    print(f"{'='*80}")
    print("✅ 주요 기능:")
    print("  1. ✅ LangGraph 기반 에이전트 그래프")
    print("  2. ✅ EXAONE 4.0-32B-AWQ Reasoning 모델")
    print("  3. ✅ vLLM 서버 기반 고속 추론")
    print("  4. ✅ LangChain 기반 RAG 시스템")
    print("  5. ✅ FAISS 벡터 데이터베이스")
    print("  6. ✅ Mistral OCR + 자동 캐싱")
    print("  7. ✅ 실시간 스트리밍 응답")
    print(f"{'='*80}")
    print("🔧 그래프 구조:")
    print("  START → Agent ─┬─→ Tools → END")
    print("                  ├─→ Retrieve → END")
    print("                  ├─→ Report → END")
    print("                  └─→ END")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f'{PROJECT_NAME} {VERSION} (LangGraph)')
    parser.add_argument('--host', type=str, default=GRADIO_HOST, help='Host IP address')
    parser.add_argument('--port', type=int, default=GRADIO_PORT, help='Port number')
    parser.add_argument('--share', action='store_true', default=True, help='Create a public URL')
    parser.add_argument('--no-share', dest='share', action='store_false', help='Disable public URL')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print_startup_info()
    print(f"📍 접속 URL: http://localhost:{args.port}")
    print(f"🤖 vLLM 서버: {VLLM_SERVER_URL}")

    try:
        app = LangGraphGradioApp()
        app.launch(host=args.host, port=args.port, share=args.share)
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"❌ 애플리케이션 실행 실패: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 문제 해결 방법:")
        print("  1. vLLM 서버가 실행 중인지 확인하세요.")
        print("  2. langgraph 패키지가 설치되어 있는지 확인하세요: pip install langgraph")
        print("  3. PDF 파일이 data/ 디렉토리에 있는지 확인하세요.")
    finally:
        try:
            app.temp_file_manager.cleanup_temp_files()
            logger.info("🗑️ 정리 작업 완료")
        except:
            pass