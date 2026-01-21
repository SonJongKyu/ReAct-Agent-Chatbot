"""
(260120) LLM 및 Embeddings 설정
"""
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

def get_json_llm():
    """JSON 형식 강제 LLM (도구 선택/쿼리 재작성용)"""
    return ChatOllama(
        model="timHan/llama3korean8B4QKM:latest", 
        format="json", 
        temperature=0
    )

def get_llm():
    """일반 LLM (최종 답변 생성용)"""
    return ChatOllama(
        model="timHan/llama3korean8B4QKM:latest", 
        temperature=0
    )

def get_embeddings():
    """Embeddings 모델"""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# 싱글톤 인스턴스
_json_llm = None
_llm = None
_embeddings = None

def get_or_create_json_llm():
    global _json_llm
    if _json_llm is None:
        _json_llm = get_json_llm()
    return _json_llm

def get_or_create_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm

def get_or_create_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings