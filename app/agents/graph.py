"""
(260120) Docstring for ChatBot.app.agents.graph
"""
"""LangGraph 그래프 구성"""
from langgraph.graph import StateGraph, END
from app.memory.state import AgentState
from app.memory.checkpointer import get_or_create_checkpointer
from app.agents.nodes import (classify_node, direct_answer_node, rewrite_node, agent_node, final_answer_node)
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