from langchain.tools import BaseTool as LangChainBaseTool
from langchain_community.vectorstores import FAISS
from abc import abstractmethod
from typing import Optional

class BaseVectorSearchTool(LangChainBaseTool):
    """벡터 검색 도구의 공통 베이스 클래스"""
    
    embeddings: Optional[object] = None
    vectorstore: Optional[FAISS] = None
    db_path: str = ""
    
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings
        self.vectorstore = None
    
    def _run(self, query: str) -> str:
        """벡터 검색 실행 (동기)"""
        if self.vectorstore is None:
            self.vectorstore = FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        
        docs = self.vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
        return "\n".join([d.page_content for d in docs])
    
    async def _arun(self, query: str) -> str:
        """벡터 검색 실행 (비동기) - 필요시 구현"""
        raise NotImplementedError("Async not implemented")