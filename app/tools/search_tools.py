"""
(테스트) 메뉴얼 DB로 가서 검색해오는 tools입니다.
여기에 새로 추가하면 __init__.py에도 추가해주세요. (호출 시 용이)
"""

from .base import BaseVectorSearchTool

from typing import Type
from pydantic import BaseModel, Field
import json, re


class SearchInput(BaseModel):
    query: str = Field(..., description="검색할 사용자 질문")


class SearchManualTool(BaseVectorSearchTool):
    """시스템 이용 방법, 매뉴얼 검색 도구"""

    name: str = "search_manual"
    description: str = "시스템 이용 방법, 매뉴얼, 이용 가이드와 관련된 질문일 때 이 도구를 사용하세요."
    args_schema: Type[BaseModel] = SearchInput
    db_path: str = "./app/faiss_db/db_total_manual"


class SearchGiftTool(BaseVectorSearchTool):
    """온누리상품권 관련 정보 검색 도구"""

    name: str = "search_gift"
    description: str = "온누리상품권 결제, 환불, 가맹점, 카드 등록과 관련된 질문일 때 이 도구를 사용하세요."
    args_schema: Type[BaseModel] = SearchInput
    db_path: str = "./app/faiss_db/db_gift"


class SearchMarketLawTool(BaseVectorSearchTool):
    """전통시장 법령 검색 도구"""

    name: str = "search_market_law"
    description: str = "전통시장 법령, 규정, 법적 근거와 관련된 질문일 때 이 도구를 사용하세요."
    args_schema: Type[BaseModel] = SearchInput
    db_path: str = "./app/faiss_db/db_market_law"

    def _run(self, query: str):
        data = super()._run(query)

        articles = []
        for r in data["results"]:
            matches = re.findall(r"(제\d+조의?\d*)", r["content"])
            articles.extend(matches)

        data["articles"] = list(dict.fromkeys(articles))
        return data

