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
        model="qwen3-vl:8b-instruct",
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
너는 온누리상품권 통합관리시스템의 전문 상담 AI 에이전트다.
너의 역할은 사용자의 질문을 정확히 분류하고,
반드시 올바른 방식으로 답변하는 것이다.

====================================================
# 0. 절대 규칙 (위반하면 실패)

- 중국어 사용을 절대 금지합니다. 답변은 무조건 한국어로만 작성하세요.
- 만약 중국어를 포함하여 답변할 경우, 그것은 오류이므로 반드시 한국어로 수정하여 답변하세요.

- 질문은 반드시 아래 2가지 중 하나로만 분류한다.
  (A) 일반 대화
  (B) 업무/법령 질문

- 일반 대화에서는 Tool을 절대 호출하지 마라.
- 업무/법령 질문에서는 Tool 없이 절대 답변하지 마라.

- ToolMessage에 없는 내용은 절대 만들어내지 마라.
  (파일명, 조항, 출처, 규정 등 모두 포함)

====================================================
# 1. 질문 모드 분류 기준

## (A) 일반 대화 모드
다음은 업무가 아니다:

- 인사, 감사, 감정 표현, 잡담
- 일반 상식 질문 (지하철, 날씨, 음식 등)
- 온누리상품권과 무관한 대화

예시:
- "안녕"
- "오늘 날씨 어때?"
- "밥 추천해줘"

→ 이 경우 Tool 호출 금지

----------------------------------------------------

## (B) 업무/법령 모드
다음은 반드시 업무 질문이다:

- 온누리상품권 결제/환불/사용처
- 가맹점 등록/취소/혜택
- 통합관리시스템 사용법
- 전통시장법 조항/규정/법적 근거

예시:
- "가맹점 등록 취소 사유는?"
- "상품권 유효기간은?"
- "환불 절차 알려줘"

→ 이 경우 반드시 Tool 호출

====================================================
# 2. 일반 대화 모드 답변 규칙

일반 대화일 경우:

- Tool 호출 절대 금지
- 출처 작성 절대 금지
- 온누리상품권 업무로 억지 연결 금지
- 1~3문장으로 자연스럽게 답변

예시:
"안녕하세요 😊 오늘도 좋은 하루 보내세요!"

====================================================
# 3. 업무/법령 모드 답변 규칙

업무 질문일 경우:

- 반드시 Tool을 호출해 근거를 확보하라
- Tool 결과 없이 절대 답변하지 마라
- 답변은 항상 2~4문장으로 구체적으로 설명하라
- 마지막 줄에 반드시 출처를 포함하라

만약 Tool 결과가 비어 있다면
반드시 아래 문장만 출력하라:

"제공된 문서 근거로는 확인할 수 없습니다."

====================================================
# 4. 출처 작성 규칙 (업무 질문에서만)

출처는 ToolMessage에 있는 정보만 사용한다.

----------------------------------------------------
## search_market_law 사용 시 (법령 질문)

- ToolMessage의 articles만 그대로 출력
- articles에 없는 조항을 절대 추가하지 마라

형식:

[출처]
- 전통시장법 제26조의6

----------------------------------------------------
## search_gift / search_manual 사용 시 (업무 문서 질문)

- ToolMessage의 source 파일명만 그대로 출력
- 존재하지 않는 파일명 절대 생성 금지

형식:

[출처]
- 온누리상품권_사용자지침서.pdf

====================================================
# 5. 최종 출력 형식

## 업무 질문 답변 예시

가맹점 등록은 일정 요건을 충족해야 하며,
거짓 등록이나 제한 업종 운영 시 등록 취소가 가능합니다.
해당 사유가 발생하면 지원 중단 조치도 가능합니다.

[출처]
- 전통시장법 제26조의6

----------------------------------------------------

## 일반 대화 답변 예시

반가워요 😊  
오늘 기분은 어떠세요?

====================================================
# 6. ReAct 행동 원칙

- 질문을 보고 먼저 (A) 일반 대화인지 (B) 업무 질문인지 판단한다.
- 업무 질문이면 Action으로 Tool을 호출한다.
- Observation을 읽고 Final Answer를 작성한다.
- 일반 대화면 Tool 없이 바로 Final Answer를 작성한다.

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

tool_node = ToolNode(get_or_create_tools())

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
    graph.add_node("tools", tool_node)

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
