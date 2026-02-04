"""
FAISS 기반 RAG 벡터 데이터베이스 생성 스크립트

기능:
1. PDF/Text 문서를 로드
2. 문서를 Chunk 단위로 분할
3. BAAI/bge-m3 임베딩 모델로 벡터화
4. 문서별 독립적인 FAISS DB 저장

사용법:
~/ChatBot 경로에서
python -m app.agents.vector_store
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ============================================================
# 1. 설정값 (Chunk / Embedding Model)
# ============================================================
MODEL_NAME = "BAAI/bge-m3"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

BASE_SAVE_PATH = "./app/faiss_db/"
# ============================================================
# 2. Embedding 모델 생성
# ============================================================
def get_embeddings():
    """
    RAG 검색용 Embedding 모델 생성
    """
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# ============================================================
# 3. 문서 로더 선택
# ============================================================
def load_documents(file_path: str):
    """
    파일 확장자에 따라 Loader 선택 후 문서 로드
    """
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    return loader.load()

# ============================================================
# 4. Chunk 분할
# ============================================================
def split_documents(documents):
    """
    문서를 Chunk 단위로 분할
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)

# ============================================================
# 5. VectorStore 생성 및 저장
# ============================================================
def create_vector_store(file_path: str, save_path: str, embeddings):
    """단일 문서 → FAISS DB 생성 후 저장"""

    print(f"\n[INFO] Processing file: {file_path}")

    # 1) 문서 로드
    documents = load_documents(file_path)

    # 2) Chunk 분할
    chunks = split_documents(documents)
    print(f"[INFO] Chunk count: {len(chunks)}")

    # 3) FAISS DB 생성
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4) 저장
    vectorstore.save_local(save_path)
    print(f"[✓] Vector store saved at: {save_path}")

# ============================================================
# 6. 실행부
# ============================================================
if __name__ == "__main__":

    # 저장 폴더 생성
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)

    # 임베딩 모델 준비
    embeddings = get_embeddings()

    # 문서 목록 정의
    data_map = {
        "gift": "./app/raw_data/온누리상품권_사용자지침서.pdf",
        "market_law": "./app/raw_data/전통시장법.pdf",
        # "total_manual": "./app/raw_data/통합관리시스템_메뉴얼.pdf",
    }

    print("\n======================================")
    print(" FAISS Vector Store Builder Started")
    print("======================================")

    for target, path in data_map.items():

        if not os.path.exists(path):
            print(f"[⚠️ SKIP] File not found: {path}")
            continue

        save_dir = os.path.join(BASE_SAVE_PATH, f"db_{target}")
        create_vector_store(path, save_dir, embeddings)

    print("\n======================================")
    print(" Vector Store Build Complete ✅")
    print("======================================")