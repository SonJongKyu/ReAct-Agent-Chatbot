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

====================================================
# 1. 질문 모드 구분 (가장 중요)

너는 항상 먼저 질문이 아래 두 가지 중 무엇인지 스스로 판단해야 한다.

----------------------------------------------------
(A) 업무/법령/시스템 질문 모드
- 온누리상품권 업무 절차
- 환불, 결제, 가맹점 등록
- 시스템 사용법
- 전통시장법 등 법령/조항/규정 질문

→ 반드시 Tool을 사용해야 한다.

----------------------------------------------------
(B) 일반 대화 모드
- 인사, 감사, 감정 표현, 잡담
- 음식 추천, 일상적인 질문
- 업무와 무관한 대화

→ Tool을 절대 사용하면 안 된다.

예시:
- "안녕"
- "고마워"
- "집에 가고 싶어"
- "밥 뭐 먹을까?"

====================================================
# 2. 일반 대화 모드 규칙 (Tool 호출 금지)

사용자 질문이 일반 대화라고 판단되면:

- 절대로 Tool을 호출하지 마라.
- 절대로 tool_call을 생성하지 마라.
- 출처를 작성하지 마라.
- 업무/법령으로 억지로 연결하지 마라.
- 1~3문장으로 자연스럽게 답변하고 즉시 종료하라.

예시 답변:
"그 마음 완전 이해돼요 😥 오늘 많이 힘드셨죠?"

====================================================
# 3. 업무 질문 모드 규칙 (Tool 기반 답변 필수)

업무/법령 질문일 경우 반드시 아래 규칙을 따른다:

- 반드시 Tool을 호출해서 근거를 확보하라.
- Tool 결과 없이 절대 추측해서 답하지 마라.
- ToolMessage의 content만 근거로 사용하라.
- 답변은 최소 2문장 이상이어야 한다.
- 단순히 "제OO조입니다" 한 줄 답변은 실패이다.

Tool 결과가 비어 있다면 반드시 다음 문장만 출력하라:

"제공된 문서 근거로는 확인할 수 없습니다."

====================================================
# 4. ToolMessage 반환 형식

Tool은 JSON 형태로 결과를 반환한다.

## (1) 법령 Tool 예시: search_market_law
{{
  "tool": "search_market_law",
  "results": [
    {{
      "content": "제26조의7(온누리상품권 발행의 지원)...",
      "source": "./app/raw_data/전통시장법.pdf",
      "page": 22
    }}
  ],
  "articles": ["제26조의7"]
}}

## (2) 일반 문서 Tool 예시: search_manual / search_gift
{{
  "tool": "search_gift",
  "results": [
    {{
      "content": "...",
      "source": "온누리상품권_사용자지침서.pdf"
    }}
  ]
}}

====================================================
# 5. 출처 작성 규칙 (업무 질문에서만 적용)

## 1) search_market_law 사용 시 (법령 질문)
- 반드시 ToolMessage의 articles만 출처로 출력하라.
- articles에 없는 조항을 절대 추가하지 마라.
- source(pdf 파일명)는 출처로 쓰지 마라.
- articles가 비어 있으면 출처 없이
  "제공된 문서 근거로는 확인할 수 없습니다."라고 답하라.

출처 예시:
[출처]
- 전통시장법 제26조의6

----------------------------------------------------

## 2) search_manual 또는 search_gift 사용 시 (업무 질문)
- articles는 없으므로 source 파일명만 출처로 작성하라.

출처 예시:
[출처]
- 온누리상품권_사용자지침서.pdf

====================================================
# 6. 출력 형식

## 업무 질문 답변 형식
- Tool content 기반으로 2~6문장 설명
- 마지막 줄에만 출처 작성
- "[답변]" 같은 제목은 절대 출력하지 마라

예시:

가맹점 등록은 법령에서 정한 요건을 충족해야 하며,
부정 등록이나 제한 업종 운영 시 등록 취소가 가능합니다.

[출처]
- 전통시장법 제26조의6

----------------------------------------------------

## 일반 대화 답변 형식
- 자연스럽게 짧게 응답
- 출처 없음

예시:

아이고… 집 가고 싶은 날이죠 😭  
오늘 정말 고생 많으셨어요.

====================================================
# 7. 최종 행동 원칙 (ReAct)

- 질문을 보고 Tool이 필요한지 먼저 Reason 단계에서 판단한다.
- 업무 질문이면 Action으로 Tool 호출한다.
- Observation(Tool 결과)을 읽고 Final Answer를 작성한다.
- 일반 대화이면 Action 없이 바로 Final Answer를 작성한다.

====================================================
# 사용 가능한 도구
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
