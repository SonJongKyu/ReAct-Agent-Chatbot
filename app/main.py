"""
Agent 기반 RAG 챗봇 메인 실행 파일
"""
import os 
import sys
from pathlib import Path


# # 1. 현재 main.py 파일이 있는 폴더(ChatBot/)의 절대 경로
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # 2. 파이썬이 모듈을 찾을 때 이 폴더를 가장 먼저 뒤지도록 설정
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)
# 


# 1. 현재 파일(main.py)의 부모의 부모 폴더(ChatBot/) 경로를 가져옵니다.
# Path(__file__).resolve()는 main.py의 절대경로
# .parent는 app/ 폴더
# .parent.parent는 ChatBot/ 폴더입니다.
root_dir = str(Path(__file__).resolve().parent.parent)

# 2. 프로젝트 루트를 sys.path에 추가
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from langchain_core.messages import HumanMessage
from app.agents.graph import get_or_create_app
from app.agents.nodes import get_or_create_tools

def run_rag_agent(user_question: str, user_id: str = "default"):
    """RAG Agent 실행"""
    app = get_or_create_app()
    config = {"configurable": {"thread_id": user_id}}
    
    # ===== 기존 대화 이력 가져오기 =====
    try:
        # 체크포인터에서 기존 상태 가져오기
        existing_state = app.get_state(config)
        previous_messages = existing_state.values.get("messages", []) if existing_state else []
    except:
        previous_messages = []
    
    # 새 사용자 메시지 추가
    new_message = HumanMessage(content=user_question)
    
    inputs = {
        "question": user_question,
        "messages": previous_messages + [new_message],  # 기존 이력 + 새 질문
        "queries": [],
        "tool_results": {},
        "final_answer": "",
        "user_id": user_id
    }
    
    result = app.invoke(inputs, config=config)
    
    metadata = {
        "queries": result.get("queries", []),
        "tool_results_keys": list(result.get("tool_results", {}).keys())
    }
    
    return result["final_answer"], metadata

if __name__ == "__main__":
    tools_dict = get_or_create_tools()
    
    print("=== 프롬프트 기반 RAG 챗봇 (종료: '종료') ===")
    print("\n사용 가능한 도구:")
    for name, tool in tools_dict.items():
        print(f"  - {name}: {tool.description}")
    print()
    
    # 사용자 ID 입력 (또는 자동 생성)
    user_id = input("사용자 ID (Enter로 기본값): ").strip() or "default"
    print(f"사용자 ID: {user_id}")
    
    while True:
        user_q = input("\n나: ")
        if user_q.strip() in ["종료", "exit", "quit"]:
            print("챗봇을 종료합니다.")
            break
        
        if not user_q.strip():
            continue
            
        try:
            final_ans, meta = run_rag_agent(user_q, user_id=user_id)
            
            print("\n" + "=" * 60)
            print(f"[사용된 쿼리]: {', '.join(meta['queries'])}")
            print(f"[사용된 도구]: {', '.join(meta['tool_results_keys'])}")
            print("=" * 60)
            print(f"\n[답변]\n{final_ans}")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n[오류 발생]: {str(e)}")
            import traceback
            traceback.print_exc()