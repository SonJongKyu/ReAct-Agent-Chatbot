---
title: ChatBot
app_file: app.py
sdk: gradio
sdk_version: 6.4.0
---
````markdown
# 🤖 ChatBot  
통합관리시스템 챗봇 서비스 개발

---

## 📦 Version
- **Python**: 3.12.1  

---

## 🌿 Branch: `temp2`
- Tool-calling 미지원 모델 로컬 사용  
  - `timHan/llama3korean8B4QKM:latest`
- 단일 기능 에이전트  
  - 질문 시 적절한 DB를 탐색하여 답변하는 에이전트

---

## 🧩 내부 구성

```text
ChatBot/
├── app/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── agent.py          # 통합 에이전트
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── search_tools.py
│   ├── faiss_db/
│   │   ├── db_gift.py
│   │   └── db_market_law.py
│   ├── raw_data/
│   │   ├── 온누리상품권_사용자지침서.py
│   │   └── 전통시장법.py
│   ├── __init__.py
│   └── main.py
````

---

## 🛠️ 현재 구성 요소

* `main.py`
* `agent.py`
* `tools`

### 📄 `agent.py` 역할

1. LLM 및 임베딩 모델 정의
2. State 정의
3. Tools 호출
4. Prompt 정의
5. Node 정의
6. Checkpointer 정의
7. Graph 정의

---

## ▶️ 사용 방법

### 1️⃣ DB 벡터화 (최초 1회 실행)

`vector_store.py` 실행 시
`faiss_db/` 폴더 하위에 각 파일별 `db_xxx` 폴더가 생성됩니다.

> ⚠️ 사전 준비
>
> * `raw_data/` 폴더에 매뉴얼 파일 (PDF 등)을 추가해야 합니다.

---

### 2️⃣ 챗봇 실행

```bash
python main.py
```

* 실행 후 대화가 시작됩니다.

```
```
