# """
# (260120) Docstring for ChatBot.app.memory.checkpoint
# """
# from langgraph.checkpoint.sqlite import SqliteSaver # 체크포인터 (프로덕션에서는 PostgreSQL이 유리)
# import sqlite3
# import os

# _checkpointer = None

# def get_or_create_checkpointer():
#     global _checkpointer
#     if _checkpointer is None:
#         # 프로젝트 루트에 database.db 파일 생성
#         db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'database.db')
        
#         # SQLite 연결 생성
#         conn = sqlite3.connect(db_path, check_same_thread=False)
        
#         # SqliteSaver 인스턴스 생성
#         _checkpointer = SqliteSaver(conn)
        
#         # 테이블 생성
#         _checkpointer.setup()
        
#         print(f"[✓] SQLite 체크포인트 DB 생성: {db_path}")
    
#     return _checkpointer
