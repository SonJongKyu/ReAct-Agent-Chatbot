"""
Agent 기반 RAG 챗봇 - 통합 파일
"""
# ============================================================
# 1. LLM 및 Embeddings 설정
# ============================================================

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings


def get_json_llm():
    """JSON 형식 강제 LLM (도구 선택/쿼리 재작성용)"""
    return ChatOllama(
        model="timHan/llama3korean8B4QKM:latest", 
        format="json", 
        temperature=0
    )

def get_llm():
    """일반 LLM (최종 답변 생성용)"""
    return ChatOllama(
        model="timHan/llama3korean8B4QKM:latest", 
        temperature=0
    )

def get_embeddings():
    """Embeddings 모델"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# 싱글톤 인스턴스
_json_llm = None
_llm = None
_embeddings = None

def get_or_create_json_llm():
    global _json_llm
    if _json_llm is None:
        _json_llm = get_json_llm()
    return _json_llm

def get_or_create_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm

def get_or_create_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings



# ============================================================
# 2. State 정의
# ============================================================

from typing import TypedDict, List, Dict, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """Agent의 상태 정의"""
    # 기본 필드
    question: str
    messages: Annotated[List[BaseMessage], operator.add]
    user_id: str
    
    # Query rewriting
    queries: List[str]
    
    # Tool execution
    tool_results: Dict[str, str]
    
    # Final answer
    final_answer: str
    
    # 분류 관련
    question_type: str
    classification_reason: str

# ============================================================
# 3. Tools 호출
# ============================================================

from app.tools import SearchManualTool, SearchGiftTool, SearchMarketLawTool

# 도구 초기화 (한 번만 생성)
def get_tools_dict():
    embeddings = get_or_create_embeddings()
    return {
        # "search_manual": SearchManualTool(embeddings),
        "search_gift": SearchGiftTool(embeddings),
        "search_market_law": SearchMarketLawTool(embeddings)
    }

_tools_dict = None

def get_or_create_tools():
    global _tools_dict
    if _tools_dict is None:
        _tools_dict = get_tools_dict()
    return _tools_dict




# ============================================================
# 4. 프롬프트 정의
# ============================================================

from langchain_core.prompts import ChatPromptTemplate


REWRITE_PROMPT = ChatPromptTemplate.from_template("""
당신은 질문 재작성 전문가입니다. 
사용자의 질문을 분석하여 검색에 최적화된 3개의 쿼리로 재작성하세요.

반드시 아래 JSON 형식으로만 답변하세요:
{{
    "queries": ["재작성된 질문1", "재작성된 질문2", "재작성된 질문3"]
}}

사용자 질문: {question}
""")

# === agent(:도구선택) 노드용 === 
def get_tool_selection_prompt(question: str, queries: list) -> str:
    """도구 선택 프롬프트 생성"""
    return f"""당신은 도구 선택 전문가입니다.
사용자의 질문을 분석하여 어떤 도구를 사용할지 결정하세요.

사용 가능한 도구:
1. search_manual: 시스템 이용 방법, 매뉴얼, 이용 가이드
2. search_gift: 온누리상품권 결제, 환불, 가맹점, 카드 등록
3. search_market_law: 전통시장 법령, 규정, 법적 근거

검색 쿼리들: {', '.join(queries)}

반드시 아래 JSON 형식으로만 답변하세요 (1~3개 선택 가능):
{{
    "selected_tools": ["search_manual"],
    "reason": "선택 이유"
}}

사용자 질문: {question}"""

# === answer 노드용 프롬프트 === 
def get_final_answer_prompt(question: str, context: str) -> str:
    """최종 답변 프롬프트 생성"""
    return f"""당신은 온누리상품권 통합관리시스템의 전문 상담 에이전트입니다.

아래 검색 결과를 바탕으로 사용자 질문에 친절하고 정확하게 답변하세요.

검색 결과:
{context}

사용자 질문: {question}

답변 작성 시 유의사항:
1. 검색된 정보만을 기반으로 답변하세요
2. 친절하고 이해하기 쉽게 설명하세요 (단, 감사 또는 인사는 제외하세요)
3. 필요한 경우 단계별로 안내하세요
4. 검색 결과가 불충분하면 그 사실을 알려주세요
5. 추측하지 말고 확실한 정보만 제공하세요

답변:"""


# === classify 노드용 프롬프트 ===

CLASSIFY_PROMPT = ChatPromptTemplate.from_template("""당신은 사용자의 질문 유형을 분류하는 전문가입니다.

대화 이력:
{conversation_history}

현재 질문: {question}

위 정보를 바탕으로 질문을 다음 3가지 유형으로 분류하세요:

1. "direct": 검색 없이 이전 대화 내용만으로 답변 가능
   - 예: "고마워", "더 자세히 설명해줘", "방금 말한거 요약해줘", "다시 말해줘"
   - 예: "이전 답변 그대로 보여줘" 
   - 예: "이해 못했으니 다시", "못 알아들었어"
                                                   
2. "context_search": 이전 대화의 맥락을 이어가며 새로운 검색 필요
   - 예: "그럼 다른(특정) 경우에는 어때?", "한도는?", "관련 정책은?" (이전 주제 관련)
   
3. "new_search": 완전히 새로운 주제로 새 검색 필요
   - 예: "환불 규정 알려줘", "배송 정책은?" (이전과 무관한 새 주제)

**판단 기준**:
- 대화 이력이 없으면 무조건 "new_search"
- "그거", "그럼", "그것" 등 지시어가 있으면 "context_search" 가능성 높음
- 감사/확인/요약 요청은 "direct"

JSON 형식으로 응답하세요:
{{
  "type": "direct | context_search | new_search",
  "reason": "분류 이유 간단히"
}}""")


# === Direct Answer 노드용 프롬프트 ===
DIRECT_ANSWER_PROMPT = """당신은 친절한 고객 서비스 챗봇입니다.

대화 이력:
{conversation_history}

현재 질문: {question}

**지시사항**:
- 위 대화 이력을 바탕으로 사용자의 질문에 답변하세요
- 이전 답변의 내용을 참조하여 요약, 재설명, 보충 설명을 제공하세요
- 추가 정보 검색은 하지 않았으므로, 이전 대화에서 언급된 내용만 사용하세요
- "관련된 내용이 없습니다" 같은 불필요한 문구는 사용하지 마세요
- 자연스럽고 친절한 톤을 유지하세요
- 감사 인사에는 간단히만 응답하고, 추가 도움을 제안하세요

답변:"""

# ============================================================
# 5. Nodes 정의
# ============================================================

import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage



def classify_node(state: AgentState):
    """대화 이력을 보고 질문 유형을 분류하는 노드"""
    print("--- [Step 0] Classify Question Type ---")
    
    question = state["question"]
    messages = state.get("messages", [])
    
    # ===== 핵심 수정: 첫 질문은 무조건 new_search =====
    # 현재 질문을 제외한 이전 메시지만 확인
    # messages에는 현재 질문(HumanMessage)이 이미 포함되어 있으므로
    # 길이가 1이면 이전 대화가 없는 것
    if len(messages) <= 1:
        print(f"[*] Question Type: new_search (첫 질문)")
        print(f"[*] Reason: 대화 이력 없음")
        return {
            "question_type": "new_search",
            "classification_reason": "첫 질문이므로 검색 필요"
        }
    
    # 대화 이력 구성 (현재 질문 제외, 최근 10개만)
    conversation_history = ""
    # 현재 질문(마지막)을 제외한 이전 메시지들만
    recent_messages = messages[-11:-1] if len(messages) > 11 else messages[:-1]
    
    for msg in recent_messages:
        role = "사용자" if isinstance(msg, HumanMessage) else "챗봇"
        conversation_history += f"{role}: {msg.content}\n"
    
    # 대화 이력이 실제로 비어있으면 new_search
    if not conversation_history.strip():
        print(f"[*] Question Type: new_search (이력 없음)")
        print(f"[*] Reason: 대화 이력이 비어있음")
        return {
            "question_type": "new_search",
            "classification_reason": "대화 이력이 없어 검색 필요"
        }
    
    # 이력이 있는 경우에만 LLM으로 분류
    json_llm = get_or_create_json_llm()
    classify_chain = CLASSIFY_PROMPT | json_llm | JsonOutputParser()
    
    result = classify_chain.invoke({
        "question": question,
        "conversation_history": conversation_history
    })
    
    question_type = result.get("type", "new_search")
    reason = result.get("reason", "")
    
    print(f"[*] Question Type: {question_type}")
    print(f"[*] Reason: {reason}")
    
    return {
        "question_type": question_type,
        "classification_reason": reason
    }


def direct_answer_node(state: AgentState):
    """검색 없이 대화 이력만으로 답변 생성하는 노드"""
    print("--- [Step 1-A] Direct Answer (No Search) ---")
    
    question = state["question"]
    messages = state.get("messages", [])
    
    # 대화 이력 구성 (현재 질문 제외)
    conversation_history = ""
    # 현재 질문(마지막)을 제외한 최근 20개 메시지
    recent_messages = messages[-21:-1] if len(messages) > 21 else messages[:-1]
    
    for msg in recent_messages:
        role = "사용자" if isinstance(msg, HumanMessage) else "챗봇"
        conversation_history += f"{role}: {msg.content}\n"
    
    print(f"[*] Conversation History Length: {len(conversation_history)} chars")
    print(f"[*] Number of Previous Messages: {len(recent_messages)}")
    
    llm = get_or_create_llm()
    
    prompt = DIRECT_ANSWER_PROMPT.format(
        conversation_history=conversation_history,
        question=question
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    final_answer = response.content
    
    print(f"[*] Generated Direct Answer Length: {len(final_answer)} chars")
    
    return {
    "final_answer": final_answer,
    "messages": [AIMessage(content=final_answer)]  # 챗봇 답변 추가!
}

# === 기존 노드들 (수정) ===

def rewrite_node(state: AgentState):
    """질문을 재작성하는 노드 - 대화 맥락 반영"""
    print("--- [Step 1-B] Query Rewriting ---")
    
    question = state["question"]
    messages = state.get("messages", [])
    question_type = state.get("question_type", "new_search")
    
    # 맥락 기반 검색인 경우 이전 대화 참조
    context = ""
    if question_type == "context_search" and len(messages) > 1:
        recent_messages = messages[-6:-1]  # 현재 질문 직전 5개 (사용자, 챗봇 포함)
        for msg in recent_messages:
            role = "사용자" if isinstance(msg, HumanMessage) else "챗봇"
            context += f"{role}: {msg.content}\n"
    
    json_llm = get_or_create_json_llm()
    
    # 프롬프트에 맥락 추가
    if context:
        prompt_text = f"""이전 대화:
{context}

현재 질문: {question}

위 대화 맥락을 고려하여 현재 질문을 검색 쿼리로 재작성하세요.
지시문법(대명사, 지시어 등)을 구체적 명사로 바꾸세요.

JSON 형식으로 응답:
{{"queries": ["쿼리1", "쿼리2", ...]}}"""
    else:
        prompt_text = REWRITE_PROMPT.format(question=question)
    
    result = json_llm.invoke([HumanMessage(content=prompt_text)])
    
    try:
        parsed = json.loads(result.content)
        queries = parsed.get("queries", [question])
    except:
        queries = [question]
    
    print(f"[*] Queries: {queries}")
    print(f"[*] Context-aware: {bool(context)}")
    
    return {"queries": queries}

def agent_node(state: AgentState):
    """도구 선택 및 실행을 하는 통합 노드"""
    print("--- [Step 2] Agent: Tool Selection & Execution ---")
    question = state["question"]
    queries = state["queries"]
    
    json_llm = get_or_create_json_llm()
    tools_dict = get_or_create_tools()
    
    # 도구 선택
    tool_selection_prompt = get_tool_selection_prompt(question, queries)
    selection_result = json_llm.invoke([HumanMessage(content=tool_selection_prompt)])
    
    try:
        selection_data = json.loads(selection_result.content)
        selected_tools = selection_data.get("selected_tools", [])
        reason = selection_data.get("reason", "")
    except:
        selected_tools = ["search_manual"]
        reason = "파싱 실패, 기본 도구 사용"
    
    print(f"[*] Selected Tools: {selected_tools}")
    print(f"[*] Reason: {reason}")
    
    # 도구 실행
    tool_results = {}
    for tool_name in selected_tools:
        if tool_name in tools_dict:
            print(f"\n[*] Executing: {tool_name}")
            tool = tools_dict[tool_name]
            
            try:
                result = tool._run(queries[0])
                tool_results[tool_name] = result
                print(f"[*] Result length: {len(result)} chars")
            except Exception as e:
                print(f"[!] Error in {tool_name}: {str(e)}")
                tool_results[tool_name] = f"오류 발생: {str(e)}"
    
    return {"tool_results": tool_results}

def final_answer_node(state: AgentState):
    """최종 답변을 생성하는 노드"""
    print("--- [Step 3] Final Answer ---")
    question = state["question"]
    tool_results = state["tool_results"]
    
    llm = get_or_create_llm()
    
    # 컨텍스트 생성
    context = ""
    for tool_name, result in tool_results.items():
        context += f"\n\n### [{tool_name} 검색 결과]\n{result}\n"
    
    # 최종 답변 생성
    final_prompt = get_final_answer_prompt(question, context)
    response = llm.invoke([HumanMessage(content=final_prompt)])
    
    final_answer = response.content
    print(f"\n[*] Generated Answer Length: {len(final_answer)} chars")
    
    return {
    "final_answer": final_answer,
    "messages": [AIMessage(content=final_answer)]  # 챗봇 답변 추가!
}

# ============================================================
# 6. Checkpointer 정의
# ============================================================

from langgraph.checkpoint.memory import MemorySaver


_checkpointer = None

def get_or_create_checkpointer():
    """
    체크포인터 생성
    - 개발/테스트: MemorySaver (공식 지원)
    - 프로덕션: PostgreSQL 등으로 교체 가능
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
        print("[✓] Memory 체크포인터 생성")
    return _checkpointer


# ============================================================
# 7. Graph 정의
# ============================================================

from langgraph.graph import StateGraph, END
import os



def route_after_classify(state: AgentState) -> str:
    """classify 결과에 따라 라우팅"""
    question_type = state.get("question_type", "new_search")
    
    if question_type == "direct":
        return "direct_answer"
    else:
        return "rewrite"

def create_rag_graph():
    """RAG Agent 그래프 생성"""
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("classify", classify_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("answer", final_answer_node)
    
    # 엣지 연결
    workflow.set_entry_point("classify")
    
    # 조건부 라우팅
    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "direct_answer": "direct_answer",
            "rewrite": "rewrite"
        }
    )
    
    # 기존 체인
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("agent", "answer")
    
    # 종료
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("answer", END)
    
    # 체크포인터와 함께 컴파일
    checkpointer = get_or_create_checkpointer()
    app = workflow.compile(checkpointer=checkpointer)

    # 그래프 이미지 저장 (파일이 없을 때만 저장)
    if not os.path.exists("agent_graph.png"):
        try:
            graph_image = app.get_graph().draw_mermaid_png()
            with open("agent_graph.png", "wb") as f:
                f.write(graph_image)
            print("[*] 그래프 이미지가 생성되었습니다.")
        except Exception as e:
            print(f"[!] 그래프 저장 실패: {e}")
    
    return app

# 싱글톤 인스턴스
_app = None

def get_or_create_app():
    global _app
    if _app is None:
        _app = create_rag_graph()
    return _app