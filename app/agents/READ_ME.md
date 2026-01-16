# README

현재 추가하는 test_rag_agent.py는 하나의 기능을 하는 에이전트이고 기본적인 틀로 구현되면 main 브랜치의 폴더구조처럼 구조를 나눌 생각입니다. (현재는 tools만 분리함)

## run_rag_agent - 하나의 에이전트 (질문시 적절한 DB 찾아가서 답변하는 에이전트)

### 내부 구성

**1. 질문 재작성 (LCEL 체인 구조)**
- 프롬프트|LLM|파싱, 재작성된 쿼리들을 반환

**2. 에이전트 실행 (create_tool_calling_agent)**
- create_tool_calling_agent가 LLM에게 도구 목록과 설명 전달
- LLM이 쿼리를 보고 각 도구의 docstring을 참고하여 적절한 도구 선택
- AgentExecutor가 선택된 도구 실행 → 결과 수집 → 답변 생성

(* 문제점: create_tool_calling_agent를 비롯한 AgentExecutor가 버전지원이 더이상 안돼서 LangGraph를 사용하라는 얘기가 있음.. 다른 방식으로 대체 필요.. help help)

**3. Self-RAG 검증(LCEL 체인 구조)**
- 현재는 검색된 DB내용과 LLM이 생성한 답변 끼리 비교하는 식으로 검증함

(* 사용자 질문에 대한 답변이 맞는지 판단하는 검증도 추가하면 좋을 것 같음)

* 참고 사항: 2번의 문제점을 해결하면 다시 공유드리겠습니다.

### 사용방법

**1. DB 벡터화 (최초 실행)**

test_vector_store.py을 실행하면 'faiss_db' 폴더에 각 파일마다의 'db_xxx' 폴더가 생성됩니다.

* 단, 'raw_data' 폴더에 메뉴얼 파일 (pdf 등)을 넣어야 함

**2. test_rag_agent.py 를 실행하면 대화가 시작됩니다. (지금은 2번 문제로 안돌아감)**