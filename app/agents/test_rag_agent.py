"""
Agent 기반 RAG 챗봇 시스템 (Tool Calling 미지원 모델용. Prompt가 tool을 호출해서 씁니다.)
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import operator
from typing import TypedDict, Annotated, List

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage

from langgraph.graph import StateGraph, END

from app.tools import SearchManualTool, SearchGiftTool, SearchMarketLawTool

# --- 1. LLM 및 도구 설정 ---
# 도구 선택/쿼리 재작성용 (JSON 강제)
json_llm = ChatOllama(model="timHan/llama3korean8B4QKM:latest", format="json", temperature=0)

# 최종 답변 생성용 (텍스트 자유 형식)
llm = ChatOllama(model="timHan/llama3korean8B4QKM:latest", temperature=0)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 도구 인스턴스 생성
tools_dict = {
    "search_manual": SearchManualTool(embeddings),
    "search_gift": SearchGiftTool(embeddings),
    "search_market_law": SearchMarketLawTool(embeddings)
}

# --- 2. State 정의 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    question: str
    queries: List[str]
    tool_results: dict
    final_answer: str

# --- 3. 프롬프트 설정 ---
rewrite_prompt = ChatPromptTemplate.from_template("""
당신은 질문 재작성 전문가입니다. 
사용자의 질문을 분석하여 검색에 최적화된 3개의 쿼리로 재작성하세요.

반드시 아래 JSON 형식으로만 답변하세요:
{{
    "queries": ["재작성된 질문1", "재작성된 질문2", "재작성된 질문3"]
}}

사용자 질문: {question}
""")
rewrite_chain = rewrite_prompt | json_llm | JsonOutputParser()

# --- 4. 노드 정의 ---
def rewrite_node(state: AgentState):
    """질문을 재작성하는 노드"""
    print("--- [Step 1] Query Rewriting ---")
    question = state["question"]
    result = rewrite_chain.invoke({"question": question})
    queries = result.get("queries", [])
    print(f"[*] Queries: {queries}")
    return {"queries": queries}

def agent_node(state: AgentState):
    """도구 선택 및 실행을 하는 통합 노드"""
    print("--- [Step 2] Agent: Tool Selection & Execution ---")
    question = state["question"]
    queries = state["queries"]
    
    # ✅ 도구 선택 (LLM에게 물어봄)
    tool_selection_prompt = f"""당신은 도구 선택 전문가입니다.
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
    
    selection_result = json_llm.invoke([HumanMessage(content=tool_selection_prompt)])
    
    try:
        import json
        selection_data = json.loads(selection_result.content)
        selected_tools = selection_data.get("selected_tools", [])
        reason = selection_data.get("reason", "")
    except:
        # JSON 파싱 실패 시 기본값
        selected_tools = ["search_manual"]
        reason = "파싱 실패, 기본 도구 사용"
    
    print(f"[*] Selected Tools: {selected_tools}")
    print(f"[*] Reason: {reason}")
    
    # ✅ 도구 실행
    tool_results = {}
    for tool_name in selected_tools:
        if tool_name in tools_dict:
            print(f"\n[*] Executing: {tool_name}")
            tool = tools_dict[tool_name]
            
            try:
                # 첫 번째 쿼리로 검색
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
    
    # 도구 실행 결과를 컨텍스트로 정리
    context = ""
    for tool_name, result in tool_results.items():
        context += f"\n\n### [{tool_name} 검색 결과]\n{result}\n"
    
    # 최종 답변 생성 프롬프트
    final_prompt = f"""당신은 온누리상품권 통합관리시스템의 전문 상담 에이전트입니다.

아래 검색 결과를 바탕으로 사용자 질문에 친절하고 정확하게 답변하세요.

검색 결과:
{context}

사용자 질문: {question}

답변 작성 시 유의사항:
1. 검색된 정보만을 기반으로 답변하세요
2. 친절하고 이해하기 쉽게 설명하세요
3. 필요한 경우 단계별로 안내하세요
4. 검색 결과가 불충분하면 그 사실을 알려주세요
5. 추측하지 말고 확실한 정보만 제공하세요

답변:"""
    
    response = llm.invoke([HumanMessage(content=final_prompt)])
    
    final_answer = response.content
    print(f"\n[*] Generated Answer Length: {len(final_answer)} chars")
    
    return {"final_answer": final_answer}

# --- 5. 그래프 구성 ---
workflow = StateGraph(AgentState)

# ✅ 3개 노드만 사용
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("agent", agent_node)      # 도구 선택 + 실행 통합
workflow.add_node("answer", final_answer_node)

# 엣지 연결 (순차 실행)
workflow.set_entry_point("rewrite")
workflow.add_edge("rewrite", "agent")
workflow.add_edge("agent", "answer")
workflow.add_edge("answer", END)

app = workflow.compile()

# --- 6. 실행 함수 ---
def run_rag_agent(user_question, chat_history):
    """RAG Agent 실행"""
    inputs = {
        "question": user_question,
        "messages": [HumanMessage(content=user_question)],
        "queries": [],
        "tool_results": {},
        "final_answer": ""
    }
    
    result = app.invoke(inputs)
    
    metadata = {
        "queries": result.get("queries", []),
        "tool_results_keys": list(result.get("tool_results", {}).keys())
    }
    
    return result["final_answer"], metadata

# --- 7. 메인 실행 ---
if __name__ == "__main__":
    history = []
    print("=== 프롬프트 기반 RAG 챗봇 (종료: '종료') ===")
    print("\n사용 가능한 도구:")
    for name, tool in tools_dict.items():
        print(f"  - {name}: {tool.description}")
    print()
    
    while True:
        user_q = input("\n나: ")
        if user_q.strip() in ["종료", "exit", "quit"]:
            print("챗봇을 종료합니다.")
            break
        
        if not user_q.strip():
            continue
            
        try:
            final_ans, meta = run_rag_agent(user_q, history)
            
            print("\n" + "=" * 60)
            print(f"[사용된 쿼리]: {', '.join(meta['queries'])}")
            print(f"[사용된 도구]: {', '.join(meta['tool_results_keys'])}")
            print("=" * 60)
            print(f"\n[답변]\n{final_ans}")
            print("=" * 60)
            
            history.append(f"사용자: {user_q}")
            history.append(f"AI: {final_ans}")
            
        except Exception as e:
            print(f"\n[오류 발생]: {str(e)}")
            import traceback
            traceback.print_exc()