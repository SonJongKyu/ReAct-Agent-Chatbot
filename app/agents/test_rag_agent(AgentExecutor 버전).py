"""
★ 이 코드의 이슈: 앞으로 AgentExecutor 대신에 LangGraph로 개발하는 것이 레거시를 최소화하는 것이라고 하네요

Agent 기반 RAG 챗봇 시스템

주요 기능:
- 질문을 3개의 검색 쿼리로 자동 재작성 (Query Rewriting)
- LangChain Agent가 도구를 자동 선택하여 적절한 벡터 DB 검색
- Self-RAG를 통한 답변 검증
- 대화 히스토리 유지 (최근 5개)

도구(Tools):
- tools/search_tools.py에 정의했습니다. 
- search_manual: 시스템 이용 방법/매뉴얼  << 임시로 지정했습니다.
- search_gift: 온누리상품권 관련
- search_market_law: 전통시장 법령

사용법:
1. FAISS 벡터 DB가 미리 생성되어 있어야 함 (db_total_manual, db_gift, db_market_law)
2. python script.py 실행
3. 대화형으로 질문 입력
4. '종료' 입력 시 프로그램 종료
"""
import json
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor
from langchain_core.agents import create_tool_calling_agent
from tools import SearchManualTool, SearchGiftTool, SearchMarketLawTool


import os


# 1. 환경 설정
llm = ChatOllama(model="timHan/llama3korean8B4QKM:latest", format="json", temperature=0)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. 도구(Tools) 인스턴스 생성 및 LangChain Tool로 변환
manual_tool_instance = SearchManualTool(embeddings)
gift_tool_instance = SearchGiftTool(embeddings)
law_tool_instance = SearchMarketLawTool(embeddings)

# LangChain Tool 형식으로 변환 (기존 @tool 데코레이터와 동일하게 사용)
tools = [
    Tool(
        name=manual_tool_instance.name,
        description=manual_tool_instance.description,
        func=manual_tool_instance.run
    ),
    Tool(
        name=gift_tool_instance.name,
        description=gift_tool_instance.description,
        func=gift_tool_instance.run
    ),
    Tool(
        name=law_tool_instance.name,
        description=law_tool_instance.description,
        func=law_tool_instance.run
    )
]

# 3. Step 1 & 2: 질문 재작성 프롬프트
rewrite_prompt = ChatPromptTemplate.from_template("""
당신은 질문 재작성 전문가입니다. 
사용자의 모호한 질문을 분석하여 검색에 최적화된 3개의 전문적인 검색 쿼리로 재작성하세요.

반드시 아래 JSON 형식으로만 답변하세요:
{{
    "queries": ["재작성된 질문1", "재작성된 질문2", "재작성된 질문3"]
}}

사용자 질문: {question}
""")

# 4. Step 4: Self-RAG 검증 프롬프트
verify_prompt = ChatPromptTemplate.from_template("""
당신은 답변 검증관입니다. 제시된 문맥(Context)을 바탕으로 답변이 정확한지 판단하세요.
결과는 JSON으로 'score' (yes/no)와 'reason'을 포함하세요.

문맥: {context}
답변: {answer}
""")

# 5. 에이전트 프롬프트 (도구 사용 지시 포함)
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 전통시장 및 온누리상품권 전문 상담 에이전트입니다.
    다음 도구들을 활용하여 사용자의 질문에 정확하게 답변하세요:
    - search_manual: 시스템 이용 방법, 매뉴얼 관련
    - search_gift: 온누리상품권 관련
    - search_market_law: 전통시장 법령 관련
    
    주어진 검색 쿼리들을 사용하여 적절한 도구를 선택하고 검색하세요."""),
    ("human", """검색 쿼리: {queries}
    이전 대화: {history}
    원래 질문: {question}
    
    위 쿼리들을 사용하여 적절한 도구로 검색하고 종합적인 답변을 제공하세요."""),
])

def run_rag_agent(user_question, chat_history=[]):
    # [1] 질문 재작성 (LCEL)
    rewrite_chain = rewrite_prompt | llm | JsonOutputParser()
    analysis = rewrite_chain.invoke({"question": user_question})
    
    queries = analysis['queries']
    print(f"[*] 재작성된 쿼리들: {queries}")

    # [2] 에이전트 실행 (도구 자동 선택 및 활용)
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True  


    )
    
    # 에이전트가 쿼리들을 처리하고 답변 생성
    result = agent_executor.invoke({
        "queries": queries,
        "history": "\n".join(chat_history[-5:]) if chat_history else "",  # 최근 5개 대화만
        "question": user_question
    })
    
    initial_answer = result['output']
    
    # 검색된 문맥 수집 (검증용)
    # 에이전트가 사용한 도구 결과를 추출
    context = ""
    if 'intermediate_steps' in result:
        for action, observation in result['intermediate_steps']:
            context += observation + "\n"
    
    # [3] Self-RAG 검증
    if context:  # 문맥이 있을 때만 검증
        verify_chain = verify_prompt | llm | JsonOutputParser()
        verification = verify_chain.invoke({"context": context, "answer": initial_answer})
        
        if verification['score'] == 'yes':
            return initial_answer, {"queries": queries, "used_tools": "agent_auto_selected"}
        else:
            return "죄송합니다. 검색된 정보의 정확성이 부족하여 답변을 드릴 수 없습니다. 질문을 조금 더 구체화해주시겠어요?", {"queries": queries, "verification": verification}
    
    return initial_answer, {"queries": queries, "used_tools": "agent_auto_selected"}


if __name__ == "__main__":
    history = []
    print("=== 에이전트 RAG 챗봇이 시작되었습니다 (종료하려면 '종료' 입력) ===")
    
    while True:
        user_q = input("\n나: ")
        if user_q in ["종료", "exit", "quit"]:
            break
            
        final_ans, meta = run_rag_agent(user_q, history)
        
        print("-" * 30)
        print(f"[*] 메타 정보: {meta}")
        print(f"[*] 답변:\n{final_ans}")
        print("-" * 30)
        
        # 대화 맥락 유지 (Memory)
        history.append(f"사용자: {user_q}")
        history.append(f"AI: {final_ans}")