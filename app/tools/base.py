"""
FAISS 기반 Vector Search Tool 공통 베이스 클래스

역할:
- 검색 Tool들이 공통으로 사용하는 FAISS 로딩 + 검색 로직 제공
- 최초 실행 시에만 DB를 lazy loading
- ToolNode가 dict 형태로 넘기는 query 입력도 자동 처리
"""

from typing import Optional, Any, Dict

from langchain.tools import BaseTool as LangChainBaseTool
from langchain_community.vectorstores import FAISS

class BaseVectorSearchTool(LangChainBaseTool):
    """
    FAISS 기반 검색 Tool 공통 베이스 클래스

    특징:
    - 최초 실행 시에만 FAISS DB를 로드 (lazy loading)
    - ToolNode가 dict 형태로 넘기는 query도 자동 처리
    - 검색 결과를 JSON dict로 반환
    """
    
    embeddings: Optional[object] = None
    vectorstore: Optional[FAISS] = None
    
    # 상속 Tool에서 반드시 지정해야 하는 DB 경로
    db_path: str = ""

    def __init__(self, embeddings):
        """embeddings 모델을 주입받아 Tool 초기화"""
        super().__init__()
        self.embeddings = embeddings
        self.vectorstore = None

    # ============================================================
    # 1. Query Normalize
    # ============================================================
    def _normalize_query(self, query: Any) -> str:
        """ToolNode가 dict 형태로 전달을 문자열 query로 정규화"""
        
        if isinstance(query, dict):
            return (query.get("query") or "").strip()
            
        return str(query).strip()

    # ============================================================
    # 2. VectorStore Lazy Loading
    # ============================================================
    def _load_vectorstore(self):
        """FAISS DB를 최초 1회만 로드"""
        if self.vectorstore is None:
            self.vectorstore = FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            print(f"[Tool Loaded] {self.name} → {self.db_path}")

    # ============================================================
    # 3. Tool 실행
    # ============================================================
    def _run(self, query: Any) -> Dict[str, Any]:
        """검색 실행"""

        query = self._normalize_query(query)

        # 빈 query 방어
        if not query:
            return {"tool": self.name, "results": []}

        # VectorStore 준비
        self._load_vectorstore()

        # 검색 수행
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        # 결과 정리
        results = []
        for doc in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", None),
                }
            )

        return {"tool": self.name, "results": results}