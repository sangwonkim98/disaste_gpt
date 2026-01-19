"""
LangGraph 그래프 시각화

그래프 구조를 다양한 형식으로 시각화합니다:
- Mermaid 다이어그램 (텍스트)
- PNG 이미지
- ASCII 아트
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_mermaid_diagram() -> str:
    """Mermaid 다이어그램 텍스트 반환"""
    from graph.builder import build_graph

    graph = build_graph()

    try:
        # LangGraph의 내장 Mermaid 생성 기능 사용
        mermaid = graph.get_graph().draw_mermaid()
        return mermaid
    except Exception as e:
        logger.warning(f"Mermaid 생성 실패: {e}")
        # 수동으로 Mermaid 다이어그램 생성
        return """
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4A90D9', 'primaryTextColor': '#fff', 'primaryBorderColor': '#2E6BB0', 'lineColor': '#5D6D7E', 'secondaryColor': '#F5B041', 'tertiaryColor': '#58D68D'}}}%%

graph TD
    subgraph LangGraph["🤖 LangGraph 에이전트 그래프"]
        START((🚀 START))
        AGENT[🧠 Agent Node<br/>의도 파악 & 라우팅]
        TOOLS[🔧 Tools Node<br/>날씨/검색 API]
        RETRIEVE[📚 Retrieve Node<br/>RAG 검색]
        REPORT[📝 Report Node<br/>보고서 생성]
        END_NODE((🏁 END))

        START --> AGENT
        AGENT -->|tools| TOOLS
        AGENT -->|retrieve| RETRIEVE
        AGENT -->|report| REPORT
        AGENT -->|end| END_NODE
        TOOLS --> END_NODE
        RETRIEVE --> END_NODE
        REPORT --> END_NODE
    end

    style START fill:#27AE60,stroke:#1E8449,color:#fff
    style AGENT fill:#3498DB,stroke:#2980B9,color:#fff
    style TOOLS fill:#F39C12,stroke:#D68910,color:#fff
    style RETRIEVE fill:#9B59B6,stroke:#7D3C98,color:#fff
    style REPORT fill:#E74C3C,stroke:#C0392B,color:#fff
    style END_NODE fill:#95A5A6,stroke:#7F8C8D,color:#fff
"""


def draw_png(output_path: str = "graph_visualization.png") -> str:
    """그래프를 PNG 이미지로 저장"""
    from graph.builder import build_graph

    graph = build_graph()

    try:
        # pygraphviz 또는 grandalf 필요
        png_data = graph.get_graph().draw_mermaid_png()

        with open(output_path, "wb") as f:
            f.write(png_data)

        logger.info(f"✅ 그래프 이미지 저장: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"PNG 생성 실패: {e}")
        logger.info("💡 PNG 생성을 위해 다음 패키지가 필요합니다:")
        logger.info("   pip install pygraphviz")
        logger.info("   또는 pip install grandalf")
        return ""


def print_ascii_diagram():
    """ASCII 아트로 그래프 출력"""
    diagram = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🤖 LangGraph 에이전트 그래프                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║                              ┌─────────┐                                     ║
║                              │  START  │                                     ║
║                              └────┬────┘                                     ║
║                                   │                                          ║
║                                   ▼                                          ║
║                         ┌─────────────────┐                                  ║
║                         │   🧠 Agent      │                                  ║
║                         │   (의도 파악)    │                                  ║
║                         └────────┬────────┘                                  ║
║                                  │                                           ║
║            ┌─────────────────────┼─────────────────────┐                     ║
║            │                     │                     │                     ║
║            ▼                     ▼                     ▼                     ║
║   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐              ║
║   │  🔧 Tools      │   │  📚 Retrieve   │   │  📝 Report     │              ║
║   │  (날씨/검색)   │   │  (RAG 검색)    │   │  (보고서 생성) │              ║
║   └───────┬────────┘   └───────┬────────┘   └───────┬────────┘              ║
║           │                    │                    │                        ║
║           │                    │                    │                        ║
║           └────────────────────┼────────────────────┘                        ║
║                                │                                             ║
║                                ▼                                             ║
║                         ┌─────────────┐                                      ║
║                         │   🏁 END    │                                      ║
║                         └─────────────┘                                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  라우팅 조건:                                                                ║
║  • "날씨", "검색", "뉴스" → Tools                                            ║
║  • "매뉴얼", "규정", "행동요령" → Retrieve                                   ║
║  • "보고서", "생성" + 업로드 → Report                                        ║
║  • 그 외 일반 대화 → END (직접 응답)                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(diagram)
    return diagram


def save_mermaid_html(output_path: str = "graph_visualization.html") -> str:
    """Mermaid 다이어그램을 HTML 파일로 저장 (브라우저에서 바로 확인 가능)"""
    mermaid_code = get_mermaid_diagram()

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LangGraph 그래프 시각화</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        h1 {{
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }}
        .mermaid {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            max-width: 90%;
        }}
        .legend {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }}
        .legend h3 {{
            margin-top: 0;
            color: #333;
        }}
        .legend ul {{
            list-style: none;
            padding: 0;
        }}
        .legend li {{
            margin: 8px 0;
            padding: 5px 10px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>🤖 LangGraph 에이전트 그래프</h1>

    <div class="mermaid">
{mermaid_code}
    </div>

    <div class="legend">
        <h3>📋 라우팅 조건</h3>
        <ul>
            <li>🔧 <strong>Tools:</strong> "날씨", "검색", "뉴스", "지진", "특보" 등 실시간 정보</li>
            <li>📚 <strong>Retrieve:</strong> "매뉴얼", "규정", "행동요령", "절차" 등 문서 검색</li>
            <li>📝 <strong>Report:</strong> "보고서 생성" + 파일 업로드</li>
            <li>🏁 <strong>END:</strong> 일반 대화 (직접 응답)</li>
        </ul>
    </div>

    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            securityLevel: 'loose',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
    </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"✅ HTML 시각화 저장: {output_path}")
    print(f"✅ HTML 파일 생성됨: {output_path}")
    print(f"💡 브라우저에서 열어보세요: file://{Path(output_path).absolute()}")
    return output_path


def visualize_graph_structure():
    """그래프 구조 정보 출력"""
    from graph.builder import build_graph

    graph = build_graph()
    graph_obj = graph.get_graph()

    print("\n" + "="*60)
    print("📊 LangGraph 그래프 구조 분석")
    print("="*60)

    # 노드 정보
    print("\n📦 노드 (Nodes):")
    for node in graph_obj.nodes:
        print(f"  • {node}")

    # 엣지 정보
    print("\n➡️ 엣지 (Edges):")
    for edge in graph_obj.edges:
        print(f"  • {edge}")

    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    print("\n🎨 LangGraph 그래프 시각화")
    print("="*50)

    # 1. ASCII 다이어그램 출력
    print("\n[1] ASCII 다이어그램:")
    print_ascii_diagram()

    # 2. Mermaid 코드 출력
    print("\n[2] Mermaid 다이어그램 코드:")
    print("-"*50)
    print(get_mermaid_diagram())

    # 3. HTML 파일 생성
    print("\n[3] HTML 파일 생성:")
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    html_path = output_dir / "graph_visualization.html"
    save_mermaid_html(str(html_path))

    # 4. 그래프 구조 분석
    print("\n[4] 그래프 구조 분석:")
    try:
        visualize_graph_structure()
    except Exception as e:
        print(f"  (구조 분석 실패: {e})")

    # 5. PNG 생성 시도
    print("\n[5] PNG 이미지 생성 시도:")
    png_path = output_dir / "graph_visualization.png"
    result = draw_png(str(png_path))
    if not result:
        print("  (PNG 생성 건너뜀 - 의존성 없음)")
