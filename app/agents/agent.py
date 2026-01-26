"""
Agent 기반 RAG 챗봇 - 통합 파일
"""
# ============================================================
# 1. LLM 및 Embeddings 설정
# ============================================================

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings


def get_react_llm():
    """ReAct Agent용 단일 LLM"""
    return ChatOllama(
        model="llama3.1:8B",
        temperature=0
    )


def get_embeddings():
    """Embeddings 모델 (RAG Tool 전용)"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )



# 싱글톤 인스턴스
_react_llm = None
_embeddings = None


def get_or_create_react_llm():
    global _react_llm
    if _react_llm is None:
        _react_llm = get_react_llm()
    return _react_llm


def get_or_create_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings


# ============================================================
# 2. State 정의
# ============================================================

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    
# ============================================================
# 3. Tools 정의
# ============================================================

from app.tools import SearchManualTool, SearchGiftTool, SearchMarketLawTool

# ReAct Agent용 Tool 리스트
_tools = None


def get_or_create_tools():
    """ReAct Agent가 사용할 Tool 목록"""
    global _tools

    if _tools is None:
        embeddings = get_or_create_embeddings()

        _tools = [
            # SearchManualTool(embeddings),  # 필요 시 활성화
            SearchGiftTool(embeddings),
            SearchMarketLawTool(embeddings),
        ]

    return _tools




# ============================================================
# 4. ReAct 프롬프트 정의 (단일 시스템 프롬프트)
# ============================================================

from langchain_core.messages import SystemMessage


def build_react_system_prompt(tools: str) -> str:
    return f"""
너는 온누리상품권 통합관리시스템의 전문 상담 에이전트다.
너의 목표는 사용자의 질문에 대해 정확하고 근거 있는 답변을 제공하는 것이다.

## 기본 행동 원칙
- 질문에 답하기 위해 필요한 경우에만 도구를 사용한다.
- 도구를 사용했다면, 반드시 그 결과를 근거로 삼아 답변한다.
- 도구 결과만으로 질문에 충분히 답할 수 있다고 판단되면, 즉시 최종 답변을 작성한다.
- 문서나 도구 결과에서 근거를 찾을 수 없다면, 추측하지 말고 그 사실을 명확히 알린다.

## 내부 사고 지침 (출력하지 말 것)
- 질문을 보고 어떤 정보가 필요한지 먼저 판단한다.
- 현재 확보된 정보가 질문에 충분한지 스스로 점검한다.
- 도구 결과 중 답변에 직접적으로 필요한 핵심 근거만 선별한다.
- 불필요한 반복, 장황한 설명, 추측성 보완 설명은 피한다.

## 최종 답변 작성 지침
- 법령·규정 질문의 경우, 관련 조항의 핵심 취지와 제도적 의미를 중심으로 설명한다.
- 사용자는 전문가가 아닐 수 있으므로, 정확성을 유지하되 이해하기 쉬운 문장으로 작성한다.
- 답변 내용은 반드시 도구를 통해 확인된 정보에 기반해야 한다.
- 감사 인사, 사족, 불필요한 서론은 포함하지 않는다.

## 사용 가능한 도구
{tools}
""".strip()




def create_system_message() -> SystemMessage:
    tools = get_or_create_tools()
    tools_str = "\n".join(
        f"- {tool.name}: {tool.description}"
        for tool in tools
    )
    return SystemMessage(content=build_react_system_prompt(tools_str))

# ============================================================
# 5. ReAct Node 정의 (단일 Loop)
# ============================================================

from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import AIMessage

def agent_node(state: AgentState):
    """
    LLM 판단 노드 (Reason 단계)
    - Tool 실행하지 않음
    - tool_calls 생성 여부만 판단
    """
    print("--- [Agent] Decide Step ---")

    llm = get_or_create_react_llm()
    tools = get_or_create_tools()

    llm_tools = [convert_to_openai_tool(t) for t in tools]

    response: AIMessage = llm.invoke(
        state["messages"],
        tools=llm_tools,
    )

    return {
        "messages": [response]
    }

# ToolNode
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(
    tools=get_or_create_tools()
)

# 종료 판단 함수 (무한 루프 차단)
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # tool_calls 없으면 최종 답변 → 종료
    if not getattr(last_message, "tool_calls", None):
        return "end"

    return "continue"


# ============================================================
# 6. Checkpointer 정의
# ============================================================

from langgraph.checkpoint.sqlite import SqliteSaver # 체크포인터 (프로덕션에서는 PostgreSQL이 유리)
import sqlite3
import os

_checkpointer = None

def get_or_create_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        # 프로젝트 루트에 database.db 파일 생성
        db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'database.db')
        
        # SQLite 연결 생성
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # SqliteSaver 인스턴스 생성
        _checkpointer = SqliteSaver(conn)
        
        # 테이블 생성
        _checkpointer.setup()
        
        print(f"[✓] SQLite 체크포인트 DB 생성: {db_path}")
    
    return _checkpointer


# ============================================================
# 7. Graph 정의
# ============================================================

import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


def create_rag_graph():
    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("agent", agent_node)
    graph.add_node(
        "tools",
        ToolNode(get_or_create_tools())
    )

    # 시작점
    graph.set_entry_point("agent")

    # agent → tool 여부 판단
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    # tool 실행 후 다시 agent
    graph.add_edge("tools", "agent")

    # 컴파일 (체크포인터 포함)
    app = graph.compile(
        checkpointer=get_or_create_checkpointer()
    )

    # 그래프 이미지 저장 (최초 1회)
    graph_image_path = "agent_graph.png"

    if not os.path.exists(graph_image_path):
        try:
            graph_image = app.get_graph().draw_mermaid_png()
            with open(graph_image_path, "wb") as f:
                f.write(graph_image)
            print("[*] Agent 그래프 이미지가 생성되었습니다.")
        except Exception as e:
            print(f"[!] 그래프 이미지 저장 실패: {e}")

    return app


# ============================================================
# 8. App Singleton (외부 진입점)
# ============================================================

_app = None

def get_or_create_app():
    """
    LangGraph App을 싱글톤으로 생성/반환
    main.py에서 이 함수만 호출하면 된다.
    """
    global _app
    if _app is None:
        _app = create_rag_graph()
    return _app


__all__ = [
    "get_or_create_app",
    "get_or_create_tools",
    "build_react_system_prompt",
]
