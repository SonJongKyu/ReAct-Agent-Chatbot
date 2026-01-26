"""
RAG 기반 LangGraph Agent 메인 실행 파일
- LangGraph 철학을 해치지 않는 관찰용 로그 포함
"""

import sys
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.agent import (
    get_or_create_app,
    get_or_create_tools,
    build_react_system_prompt,
)

# 프로젝트 루트 경로 설정
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


# ReAct Agent 실행 함수
def run_rag_agent(user_question: str, user_id: str = "default"):
    """
    LangGraph Agent 실행
    - State는 messages만 사용
    - SystemMessage는 세션 최초 1회만 주입
    - 종료 판단 / Tool 실행은 Graph에서 처리
    - main.py는 '관찰자' 역할만 수행
    """

    app = get_or_create_app()
    config = {"configurable": {"thread_id": user_id}}

    # 기존 상태 복원
    try:
        state = app.get_state(config)
        messages = list(state.values["messages"]) if state else []
    except Exception:
        messages = []

    # 최초 실행 시 System Prompt 주입
    if not messages:
        tools = get_or_create_tools()
        tools_desc = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in tools
        )
        system_prompt = build_react_system_prompt(tools_desc)

        messages.append(SystemMessage(content=system_prompt))

        print("\n[DEBUG] SystemMessage injected (first turn)")

    # 사용자 질문 추가
    messages.append(HumanMessage(content=user_question))

    # Graph 실행 전 입력 상태 로그
    print("\n[DEBUG] ===== Graph Invoke Input =====")
    for idx, msg in enumerate(messages):
        msg_type = type(msg).__name__
        preview = str(msg.content).replace("\n", " ")[:120]
        print(f"[{idx}] {msg_type}: {preview}")

    # Graph 실행
    result = app.invoke(
        {"messages": messages},
        config=config,
    )

    # Graph 실행 결과 로그
    print("\n[DEBUG] ===== Graph Result Messages =====")
    for idx, msg in enumerate(result["messages"]):
        msg_type = type(msg).__name__
        content_preview = str(msg.content).replace("\n", " ")[:200]
        print(f"[{idx}] {msg_type}: {content_preview}")

    # 최종 답변 반환
    final_message = result["messages"][-1]

    print("\n[DEBUG] ===== Final Answer Selected =====")
    print(f"Type   : {type(final_message).__name__}")
    print(f"Content: {final_message.content}")

    return final_message.content

# CLI 실행부
if __name__ == "__main__":
    print("=== LangGraph 기반 RAG 챗봇 ===")

    user_id = input("사용자 ID (Enter로 기본값): ").strip() or "default"
    print(f"사용자 ID: {user_id}")

    while True:
        user_q = input("\n나: ").strip()

        if user_q in ["종료", "exit", "quit"]:
            print("챗봇을 종료합니다.")
            break

        if not user_q:
            continue

        try:
            answer = run_rag_agent(user_q, user_id=user_id)

            print("\n" + "=" * 60)
            print("[답변]")
            print(answer)
            print("=" * 60)

        except Exception as e:
            print(f"\n[오류 발생]: {e}")
            import traceback
            traceback.print_exc()
