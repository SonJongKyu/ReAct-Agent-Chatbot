"""
(테스트) 메뉴얼 DB로 가서 검색해오는 tools입니다. 
여기에 새로 추가하면 init에도 추가해주세요. (호출 시 용이)
"""

from langchain_community.vectorstores import FAISS
from .base import BaseTool

class SearchManualTool(BaseTool):
    name = "search_manual"
    description = "시스템 이용 방법, 매뉴얼, 이용 가이드와 관련된 질문일 때 이 도구를 사용하세요."
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore = None
    
    def run(self, query: str) -> str:
        """매뉴얼 검색 실행"""
        if self.vectorstore is None:
            self.vectorstore = FAISS.load_local(
                "./faiss_db/db_total_manual", 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        
        docs = self.vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
        return "\n".join([d.page_content for d in docs])


class SearchGiftTool(BaseTool):
    name = "search_gift"
    description = "온누리상품권 결제, 환불, 가맹점, 카드 등록과 관련된 질문일 때 이 도구를 사용하세요."
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore = None
    
    def run(self, query: str) -> str:
        """상품권 정보 검색 실행"""
        if self.vectorstore is None:
            self.vectorstore = FAISS.load_local(
                "./faiss_db/db_gift", 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        
        docs = self.vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
        return "\n".join([d.page_content for d in docs])


class SearchMarketLawTool(BaseTool):
    name = "search_market_law"
    description = "전통시장 법령, 규정, 법적 근거와 관련된 질문일 때 이 도구를 사용하세요."
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore = None
    
    def run(self, query: str) -> str:
        """시장 법령 검색 실행"""
        if self.vectorstore is None:
            self.vectorstore = FAISS.load_local(
                "./faiss_db/db_market_law", 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        
        docs = self.vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
        return "\n".join([d.page_content for d in docs])