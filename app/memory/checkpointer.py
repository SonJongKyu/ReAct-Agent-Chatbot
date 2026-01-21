"""
(260120) Docstring for ChatBot.app.memory.checkpoint
"""
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

_checkpointer = None

def get_or_create_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        # 프로젝트 루트에 database.db 파일 생성
        db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'database.db')
        
        # SQLite 연결 생성
        conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # SqliteSaver 인스턴스 생성
        _checkpointer = SqliteSaver(conn)
        
        # 테이블 생성
        _checkpointer.setup()
        
        print(f"[✓] SQLite 체크포인트 DB 생성: {db_path}")
    
    return _checkpointer
# from langgraph.checkpoint.postgres import PostgresSaver
# from app.config.database import get_database_url

# def get_checkpointer():
#     """PostgreSQL 체크포인트 생성"""
#     db_url = get_database_url()
#     checkpointer = PostgresSaver.from_conn_string(db_url)
    
#     # 최초 1회 테이블 생성
#     try:
#         checkpointer.setup()
#     except Exception as e:
#         print(f"Table already exists or error: {e}")
    
#     return checkpointer

# # 싱글톤 패턴으로 재사용
# _checkpointer = None

# def get_or_create_checkpointer():
#     global _checkpointer
#     if _checkpointer is None:
#         _checkpointer = get_checkpointer()
#     return _checkpointer