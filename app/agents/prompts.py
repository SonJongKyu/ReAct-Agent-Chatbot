"""
(260120) 프롬프트 템플릿
"""


from langchain_core.prompts import ChatPromptTemplate

# === rewrite 노드용 프롬프트 === 
# node에서 LCEL로 구성되어 있음 (rewrite_chain = REWRITE_PROMPT | json_llm | JsonOutputParser())
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