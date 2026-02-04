"""
RAG 시스템에서 사용하는 FAISS 기반 검색 Tool 모음

구성:
- SearchManualTool     : 시스템 매뉴얼 검색
- SearchGiftTool       : 온누리상품권 업무 문서 검색
- SearchMarketLawTool  : 전통시장법 조항 검색

새로 추가하면 __init__.py에도 추가(호출 시 용이)
"""

import re
from typing import Type

from pydantic import BaseModel, Field

from .base import BaseVectorSearchTool

# ============================================================
# 1. 공통 입력 Schema
# ============================================================
class SearchInput(BaseModel):
    """검색 Tool 입력 형식"""
    query: str = Field(..., description="검색할 사용자 질문")

# ============================================================
# 2. 문서 검색 Tool
# ============================================================
class SearchManualTool(BaseVectorSearchTool):
    """통합관리시스템 매뉴얼 검색 Tool"""

    name: str = "search_manual"
    description: str = "시스템 이용 방법, 매뉴얼, 이용 가이드와 관련된 질문일 때 이 도구를 사용하세요."
    args_schema: Type[BaseModel] = SearchInput
    db_path: str = "./app/faiss_db/db_total_manual"


class SearchGiftTool(BaseVectorSearchTool):
    """온누리상품권 업무 문서 검색 Tool"""

    name: str = "search_gift"
    description: str = "온누리상품권 결제, 환불, 가맹점, 카드 등록과 관련된 질문일 때 이 도구를 사용하세요."
    args_schema: Type[BaseModel] = SearchInput
    db_path: str = "./app/faiss_db/db_gift"


class SearchMarketLawTool(BaseVectorSearchTool):
    """전통시장법 법령 검색 Tool"""

    name: str = "search_market_law"
    description: str = "전통시장 법령, 규정, 법적 근거와 관련된 질문일 때 이 도구를 사용하세요."
    args_schema: Type[BaseModel] = SearchInput
    db_path: str = "./app/faiss_db/db_market_law"

    def _run(self, query: str):
        """검색 실행 후 조항(article) 목록을 추가로 반환"""
        data = super()._run(query)

        articles = []
        for r in data["results"]:
            matches = re.findall(r"(제\d+조의?\d*)", r["content"])
            articles.extend(matches)

        # 중복 제거 후 추가
        data["articles"] = list(dict.fromkeys(articles))
        
        return data

