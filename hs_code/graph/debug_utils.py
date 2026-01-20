"""
Query Pipeline 디버그 유틸리티

실행 과정의 상세 로깅 및 시각화 도구
"""

import logging
import json
from typing import Any, Dict, Optional
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)


class PipelineDebugger:
    """파이프라인 실행 과정 디버깅 도구"""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.execution_log = []
        self.start_time = None

    def start(self):
        """디버깅 세션 시작"""
        self.execution_log = []
        self.start_time = datetime.now()
        if self.enabled:
            self._print_header("PIPELINE DEBUG SESSION STARTED")

    def log_node(self, node_name: str, input_data: Dict, output_data: Dict):
        """노드 실행 로그"""
        if not self.enabled:
            return

        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_sec": round(elapsed, 2),
            "node": node_name,
            "input_summary": self._summarize(input_data),
            "output_summary": self._summarize(output_data)
        }
        self.execution_log.append(entry)

        self._print_node_log(entry)

    def log_tool_call(self, tool_name: str, params: Dict, result: Any):
        """도구 호출 로그"""
        if not self.enabled:
            return

        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        print(f"""
    ╭─ 🔧 Tool Call ─────────────────────────
    │ Tool: {tool_name}
    │ Params: {json.dumps(params, ensure_ascii=False)[:100]}
    │ Result: {self._summarize(result)[:200]}
    │ Time: +{elapsed:.2f}s
    ╰────────────────────────────────────────
        """)

    def log_plan(self, plan: Dict):
        """실행 계획 로그"""
        if not self.enabled:
            return

        print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                     📋 EXECUTION PLAN                         ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║ Need Tools: {str(plan.get('need_tools', False)):<10} Tools: {str(plan.get('tool_list', [])):<25}║
    ║ Need RAG:   {str(plan.get('need_rag', False)):<10} Query: {str(plan.get('rag_query', ''))[:25]:<25}║
    ║ Need PDF:   {str(plan.get('need_pdf', False)):<10} Task:  {str(plan.get('pdf_task', '')):<25}║
    ╠═══════════════════════════════════════════════════════════════╣
    ║ Reasoning: {str(plan.get('tool_reasoning', ''))[:50]:<50} ║
    ║ Confidence: {plan.get('confidence', 0):.2f}                                           ║
    ╚═══════════════════════════════════════════════════════════════╝
        """)

    def end(self):
        """디버깅 세션 종료 및 요약"""
        if not self.enabled:
            return

        total_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   📊 EXECUTION SUMMARY                        ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║ Total Nodes Executed: {len(self.execution_log):<5}                                 ║
    ║ Total Time: {total_time:.2f}s                                           ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║ Execution Path:                                               ║""")

        for entry in self.execution_log:
            print(f"    ║   └─ {entry['node']:<20} (+{entry['elapsed_sec']:.2f}s)                  ║")

        print("""    ╚═══════════════════════════════════════════════════════════════╝
        """)

    def _print_header(self, title: str):
        """헤더 출력"""
        print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  🔍 {title:<55} ║
    ║  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<53} ║
    ╚═══════════════════════════════════════════════════════════════╝
        """)

    def _print_node_log(self, entry: Dict):
        """노드 로그 출력"""
        print(f"""
    ┌─ 📍 Node: {entry['node']:<20} (+{entry['elapsed_sec']:.2f}s) ─────────
    │ Input:  {entry['input_summary'][:60]}
    │ Output: {entry['output_summary'][:60]}
    └────────────────────────────────────────────────────────
        """)

    def _summarize(self, data: Any, max_len: int = 100) -> str:
        """데이터 요약"""
        if data is None:
            return "None"
        if isinstance(data, dict):
            keys = list(data.keys())[:5]
            return f"Dict({len(data)} keys: {keys})"
        if isinstance(data, list):
            return f"List({len(data)} items)"
        if isinstance(data, str):
            return data[:max_len] + "..." if len(data) > max_len else data
        return str(data)[:max_len]


# 전역 디버거 인스턴스
_debugger = PipelineDebugger(enabled=False)


def get_debugger() -> PipelineDebugger:
    """전역 디버거 반환"""
    return _debugger


def enable_debug():
    """디버그 모드 활성화"""
    global _debugger
    _debugger.enabled = True
    logger.info("🔍 Debug mode enabled")


def disable_debug():
    """디버그 모드 비활성화"""
    global _debugger
    _debugger.enabled = False


def debug_node(func):
    """노드 함수 디버깅 데코레이터"""
    @wraps(func)
    def wrapper(state: Dict, *args, **kwargs):
        debugger = get_debugger()

        # 입력 상태 캡처
        input_snapshot = {
            "user_input": state.get("user_input", "")[:50],
            "next_node": state.get("next_node"),
            "has_plan": bool(state.get("execution_plan")),
            "has_result": bool(state.get("execution_result"))
        }

        # 노드 실행
        output = func(state, *args, **kwargs)

        # 출력 상태 캡처
        output_snapshot = {
            "next_node": output.get("next_node") if output else None,
            "has_response": bool(output.get("final_response")) if output else False,
            "keys": list(output.keys()) if output else []
        }

        # 로깅
        debugger.log_node(func.__name__, input_snapshot, output_snapshot)

        return output

    return wrapper


# ===== 테스트용 함수 =====

def print_state_summary(state: Dict, title: str = "State"):
    """상태 요약 출력"""
    print(f"""
    ╭─ {title} ──────────────────────────────────────
    │ user_input: {state.get('user_input', '')[:40]}...
    │ next_node: {state.get('next_node')}
    │ has_plan: {bool(state.get('execution_plan'))}
    │ has_result: {bool(state.get('execution_result'))}
    │ final_response: {state.get('final_response', '')[:40]}...
    ╰──────────────────────────────────────────────────
    """)


def format_tool_results(tool_results: Dict) -> str:
    """도구 결과 포맷팅"""
    if not tool_results:
        return "No tool results"

    lines = []
    for tool_name, result in tool_results.items():
        if isinstance(result, dict):
            if "error" in result:
                lines.append(f"  ❌ {tool_name}: Error - {result['error']}")
            else:
                data_preview = str(result.get("data", result))[:100]
                lines.append(f"  ✅ {tool_name}: {data_preview}...")
        else:
            lines.append(f"  ✅ {tool_name}: {str(result)[:100]}...")

    return "\n".join(lines)
