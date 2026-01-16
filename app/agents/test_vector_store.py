"""
RAG 시스템을 위한 FAISS 벡터 데이터베이스 생성 스크립트

주요 기능:
- PDF/텍스트 문서를 로드하여 청크 단위로 분할
- BAAI/bge-m3 임베딩 모델을 사용하여 벡터화
- 각 문서별로 독립된 FAISS 벡터 DB 생성 및 저장

사용법:
1. data_map에서 문서 경로 설정
2. python test_vector_store.py 실행
3. 생성된 db_* 폴더에서 벡터 DB 로드하면 검색에 활용 가능
"""

import os
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# os.chdir("../raw_data") # 경로 유의

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 임베딩 모델 설정 (bge-m3)
model_name = "BAAI/bge-m3"
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},   
    encode_kwargs=encode_kwargs
)
def create_vector_store(file_path, save_path):
    # 파일 확장자에 따른 로더 선택
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()
    
    # 2. 청킹 (Chunking) - 독립적 처리
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # 3. FAISS DB 생성 및 저장
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(save_path)
    print(f"Vector store saved at {save_path}")


if __name__ == "__main__":
    # 1. 저장할 메인 경로 설정
    base_save_path = "./app/faiss_db/"
    
    # 폴더가 없으면 생성
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    data_map = {
        "total_manual": "./app/raw_data/통합관리시스템_메뉴얼.pdf", # 테스트 또는 적용할 파일에 맞게 수정하면 됩니다. 
        "gift": "./app/raw_data/온누리상품권_사용자지침서.pdf",
        "market_law": "./app/raw_data/전통시장법.pdf"
    }
    
    for target, path in data_map.items():
        if os.path.exists(path):
            # 2. 지정하신 경로 뒤에 파일별 이름을 붙여서 저장
            final_path = os.path.join(base_save_path, f"db_{target}")
            create_vector_store(path, final_path)