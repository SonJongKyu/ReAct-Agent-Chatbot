"""
(260120) 각 노드들(rewrite, agent, answer노드) 함수들
"""
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage  

from app.memory.state import AgentState
from app.config.llm import get_or_create_json_llm, get_or_create_llm, get_or_create_embeddings
# from app.llm.prompts import REWRITE_PROMPT, get_tool_selection_prompt, get_final_answer_prompt
from app.llm.prompts import (CLASSIFY_PROMPT, DIRECT_ANSWER_PROMPT, REWRITE_PROMPT, get_tool_selection_prompt, get_final_answer_prompt)
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

# 노드 함수들

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