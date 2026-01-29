from langchain.tools import BaseTool as LangChainBaseTool
from langchain_community.vectorstores import FAISS
from typing import Optional, Any
import json

class BaseVectorSearchTool(LangChainBaseTool):
    embeddings: Optional[object] = None
    vectorstore: Optional[FAISS] = None
    db_path: str = ""

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings
        self.vectorstore = None

    def _normalize_query(self, query: Any) -> str:
        # ToolNode가 dict/kwargs로 넘기는 경우를 흡수
        if isinstance(query, dict):
            return (query.get("query") or "").strip()
        return str(query).strip()

    def _run(self, query: str):
        query = self._normalize_query(query)

        if self.vectorstore is None:
            self.vectorstore = FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

        docs = self.vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)

        results = []
        for d in docs:
            results.append({
                "content": d.page_content,
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", None),
            })

        # ✅ dict로 반환 (중요)
        return {"tool": self.name, "results": results}
