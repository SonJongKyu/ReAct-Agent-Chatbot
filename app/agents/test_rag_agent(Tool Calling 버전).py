# """
# ★ 이 코드의 이슈: Tool Calling을 통해 툴을 호출하면 LLM 모델이 알아서 도구 자체를 인식하게 되는데 ollama는 3.3 버전만 지원된다고 하네요. (근데 ollama 3.3은 32GB 넘게 필요함)

# Agent 기반 RAG 챗봇 시스템

# 주요 기능:
# - 질문을 3개의 검색 쿼리로 자동 재작성 (Query Rewriting)
# - LangChain Agent가 도구를 자동 선택하여 적절한 벡터 DB 검색
# - Self-RAG를 통한 답변 검증
# - 대화 히스토리 유지 (최근 5개)

# 도구(Tools):
# - tools/search_tools.py에 정의했습니다. 
# - search_manual: 시스템 이용 방법/매뉴얼
# - search_gift: 온누리상품권 관련
# - search_market_law: 전통시장 법령

# 사용법:
# 1. FAISS 벡터 DB가 미리 생성되어 있어야 함 (db_total_manual, db_gift, db_market_law)
# 2. python script.py 실행
# 3. 대화형으로 질문 입력
# 4. '종료' 입력 시 프로그램 종료
# """

# import sys
# from pathlib import Path

# # 프로젝트 루트 경로 추가
# project_root = Path(__file__).resolve().parent.parent.parent
# sys.path.insert(0, str(project_root))

# import operator
# from typing import TypedDict, Annotated, List

# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage

# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode

# from app.tools import SearchManualTool, SearchGiftTool, SearchMarketLawTool

# # --- 1. LLM 및 도구 설정 ---
# llm = ChatOllama(model="timHan/llama3korean8B4QKM:latest", format="json", temperature=0)
# llm_for_agent = ChatOllama(model="timHan/llama3korean8B4QKM:latest", temperature=0)

# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-m3",
#     model_kwargs={'device': 'cpu'},
#     encode_kwargs={'normalize_embeddings': True}
# )

# # ✅ LangChain Tool 인스턴스 생성
# tools = [
#     SearchManualTool(embeddings),
#     SearchGiftTool(embeddings),
#     SearchMarketLawTool(embeddings)
# ]

# # ✅ LangChain의 ToolNode 사용 (자동으로 tool 실행)
# tool_node = ToolNode(tools)

# # ✅ LLM에 도구 바인딩 (Ollama Tool Calling)
# llm_with_tools = llm_for_agent.bind_tools(tools)

# # --- 2. State 정의 ---
# class AgentState(TypedDict):
#     messages: Annotated[List[BaseMessage], operator.add]
#     question: str
#     queries: List[str]
#     final_answer: str

# # --- 3. 프롬프트 설정 ---
# rewrite_prompt = ChatPromptTemplate.from_template("""
# 당신은 질문 재작성 전문가입니다. 
# 사용자의 질문을 분석하여 검색에 최적화된 3개의 쿼리로 재작성하세요.

# 반드시 아래 JSON 형식으로만 답변하세요:
# {{
#     "queries": ["재작성된 질문1", "재작성된 질문2", "재작성된 질문3"]
# }}

# 사용자 질문: {question}
# """)
# rewrite_chain = rewrite_prompt | llm | JsonOutputParser()

# # --- 4. 노드 정의 ---
# def rewrite_node(state: AgentState):
#     """질문을 재작성하는 노드"""
#     print("--- [Step 1] Query Rewriting ---")
#     question = state["question"]
#     result = rewrite_chain.invoke({"question": question})
#     queries = result.get("queries", [])
#     print(f"[*] Queries: {queries}")
#     return {"queries": queries}

# def agent_node(state: AgentState):
#     """LangChain Tool Calling을 사용하는 에이전트 노드"""
#     print("--- [Step 2] Agent Running ---")
#     messages = state["messages"]
#     queries = state["queries"]
    
#     # 시스템 메시지 추가
#     system_msg = SystemMessage(content=f"""당신은 온누리상품권 통합관리시스템의 전문 상담 에이전트입니다.

# 사용자를 위해 생성된 검색 쿼리: {', '.join(queries)}
# 이 쿼리들을 참고하여 적절한 도구를 사용하세요.

# 사용 가능한 도구:
# - search_manual: 시스템 이용 방법, 매뉴얼 검색
# - search_gift: 온누리상품권 관련 정보 검색
# - search_market_law: 전통시장 법령 검색

# 사용자 질문에 가장 적합한 도구를 선택하여 사용하세요.""")
    
#     if not isinstance(messages[0], SystemMessage):
#         messages = [system_msg] + messages
    
#     # ✅ Tool이 바인딩된 LLM 호출
#     response = llm_with_tools.invoke(messages)
    
#     return {"messages": [response]}

# def final_answer_node(state: AgentState):
#     """최종 답변을 생성하는 노드"""
#     print("--- [Step 3] Final Answer ---")
#     messages = state["messages"]
    
#     # 도구 실행 결과를 바탕으로 최종 답변 요청
#     final_prompt = HumanMessage(content="""위에서 검색한 정보를 바탕으로 사용자에게 친절하고 명확한 답변을 작성해주세요.

# 답변 시 유의사항:
# 1. 검색된 정보를 기반으로 정확하게 답변하세요
# 2. 친절하고 이해하기 쉽게 설명하세요
# 3. 필요한 경우 단계별로 안내하세요
# 4. 검색 결과가 없다면 그 사실을 정직하게 알려주세요""")
    
#     final_response = llm_for_agent.invoke(messages + [final_prompt])
    
#     return {"final_answer": final_response.content}

# # --- 5. 그래프 구성 ---
# def should_continue(state: AgentState):
#     """도구 호출 여부 판단"""
#     messages = state["messages"]
#     last_message = messages[-1]
    
#     # ✅ tool_calls가 있으면 도구 실행
#     if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#         print(f"[*] 도구 호출 감지: {[tc['name'] for tc in last_message.tool_calls]}")
#         return "tools"
    
#     return "answer"

# workflow = StateGraph(AgentState)

# # 노드 추가
# workflow.add_node("rewrite", rewrite_node)
# workflow.add_node("agent", agent_node)
# workflow.add_node("tools", tool_node)  # ✅ LangChain의 ToolNode 사용
# workflow.add_node("answer", final_answer_node)

# # 엣지 연결
# workflow.set_entry_point("rewrite")
# workflow.add_edge("rewrite", "agent")

# # 조건부 엣지
# workflow.add_conditional_edges(
#     "agent",
#     should_continue,
#     {
#         "tools": "tools",
#         "answer": "answer"
#     }
# )

# workflow.add_edge("tools", "agent")  # 도구 실행 후 다시 agent로
# workflow.add_edge("answer", END)

# app = workflow.compile()

# # --- 6. 실행 함수 ---
# def run_rag_agent(user_question, chat_history):
#     """RAG Agent 실행"""
#     input_messages = []
    
#     # 대화 히스토리 추가 (최근 5개)
#     if chat_history:
#         for turn in chat_history[-5:]:
#             if turn.startswith("사용자:"):
#                 input_messages.append(HumanMessage(content=turn.replace("사용자: ", "")))
#             elif turn.startswith("AI:"):
#                 input_messages.append(AIMessage(content=turn.replace("AI: ", "")))
    
#     # 현재 질문 추가
#     input_messages.append(HumanMessage(content=user_question))

#     inputs = {
#         "question": user_question,
#         "messages": input_messages,
#         "queries": [],
#         "final_answer": ""
#     }
    
#     result = app.invoke(inputs)
#     return result["final_answer"], {"queries": result["queries"]}

# # --- 7. 메인 실행 ---
# if __name__ == "__main__":
#     history = []
#     print("=== LangChain Tool 기반 RAG 챗봇 (종료: '종료') ===")
#     print("사용 가능한 도구:")
#     for tool in tools:
#         print(f"  - {tool.name}: {tool.description}")
#     print()
    
#     while True:
#         user_q = input("\n나: ")
#         if user_q.strip() in ["종료", "exit", "quit"]:
#             print("챗봇을 종료합니다.")
#             break
        
#         if not user_q.strip():
#             continue
            
#         try:
#             final_ans, meta = run_rag_agent(user_q, history)
            
#             print("-" * 50)
#             print(f"[사용된 쿼리]: {', '.join(meta['queries'])}")
#             print("-" * 50)
#             print(f"[답변]\n{final_ans}")
#             print("-" * 50)
            
#             history.append(f"사용자: {user_q}")
#             history.append(f"AI: {final_ans}")
            
#         except Exception as e:
#             print(f"[오류 발생]: {str(e)}")
#             import traceback
#             traceback.print_exc()