# 🤖 온누리상품권 업무 지원 RAG 챗봇 (LangGraph + ReAct + FAISS)

본 프로젝트는 **온누리상품권 통합관리시스템 업무 담당자**를 위한  
**RAG 기반 상담 챗봇**입니다.

LangGraph 기반의 **ReAct Agent 구조**를 적용하여

- 질문을 업무/일반 대화로 자동 분류  
- 업무 질문은 반드시 문서 검색 Tool을 호출  
- 법령/매뉴얼 근거 기반 답변만 생성  
- 사용자별 세션을 SQLite Checkpoint로 유지  

하는 것이 핵심 목표입니다.

---

## ✅ 주요 기능

- **LangGraph ReAct Agent 기반 Tool Calling**
- **FAISS 벡터 DB 기반 문서 검색 (PDF)**
- **업무 질문은 Tool 없이 답변 불가 (Hallucination 방지)**
- **법령 조항 자동 추출**
- **Gradio 기반 ChatGPT 스타일 UI**
- **사용자 ID별 대화 기록 유지 (SQLite Checkpointer)**

---

## 📌 프로젝트 구조

```bash
ChatBot/
├── README.md
├── database.db                # 사용자별 세션 저장
└── app/
    ├── main.py                # 실행 엔트리포인트 (관찰자 역할)
    ├── gradio_app.py          # Gradio UI 실행 파일
    ├── agents/
    │   ├── agent.py           # LangGraph ReAct Agent 정의
    │   └── vector_store.py    # PDF → FAISS DB 생성 스크립트
    ├── tools/
    │   ├── base.py            # 공통 벡터 검색 Tool 베이스
    │   └── search_tools.py    # 업무별 Tool 정의
    ├── raw_data/
    │   ├── ....pdf
    │   └── ....pdf
    └── faiss_db/
        ├── db_**/
        └── db_**/
```

---

## ⚙️ 실행 환경

| 역할 | 모델 |
|------|------|
| LLM (ReAct Agent) | `qwen3-vl:8b-instruct` |
| Embedding Model | `BAAI/bge-m3` |
| Vector DB | FAISS |

---

## 💻 권장 시스템 사양

본 프로젝트는 **CPU 환경에서도 실행 가능**하도록 구성되었습니다.

### 최소 사양

- CPU: 4 Core 이상  
- RAM: 16GB 이상  
- OS: Ubuntu / Windows / macOS  
- Python: 3.10 ~ 3.12  

### 권장 사양

- CPU: 8 Core 이상  
- RAM: 32GB 이상  
- GPU 없이도 가능 (속도는 느릴 수 있음)

---

## 🔧 설치 방법

### 1. 레포지토리 클론

```bash
git clone https://github.com/yourname/ChatBot.git
cd ChatBot
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 🔧 가상환경 생성 및 활성화 (Miniconda)

본 프로젝트는 **Miniconda 기반 Conda 환경**에서 실행하도록 구성되어 있습니다.

### 1. Miniconda 설치 (Linux 기준)

Miniconda 설치 가이드:  
https://www.anaconda.com/docs/getting-started/miniconda/install

```bash
mkdir -p ~/miniconda3

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -O ~/miniconda3/miniconda.sh

bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all
```
설치 후 터미널 재시작

### 2. Conda 가상환경 생성

```bash
conda create -n ChatBot python==3.12.1
```

### 3. Conda 환경 활성화

```bash
conda activate ChatBot
```
활성화되면 프롬프트가 다음과 같이 변경됩니다. 

```bash
(ChatBot) user@machine:~/ChatBot$
```

---

## 📂 문서 벡터 DB 생성

PDF 문서를 FAISS DB로 변환해야 검색이 가능합니다. 

```bash
python app/agents/vector_store.py
```

생성 후 구조:

```bash
app/faiss_db/
 ├── db_**/
 └── db_**/
```

---

## 🚀 실행 방법

### 1. CLI 실행

```bash
python -m app.main
```

### 2. Gradio 웹 UI 실행

```bash
python -m app.gradio_app
```

실행 후 브라우저에서 접속

```bash
http://localhost:7860
```

---

## 🧠 Agent 동작 방식

LangGraph 기반 단일 Loop 구조:

```bash
User Question
   ↓
Agent Node (LLM 판단)
   ↓
Tool 필요하면 ToolNode 호출
   ↓
검색 결과 Observation
   ↓
최종 답변 생성
```
업무 질문은 반드시 Tool 호출을 수행합니다.

---

## 🔍 사용 가능한 Tools

| Tool 이름 | 설명 |
|----------|------|
| `search_gift` | 온누리상품권 결제/환불/가맹점 검색 |
| `search_market_law` | 전통시장법 조항 검색 |
| `search_manual` (옵션) | 시스템 매뉴얼 검색 |

---

## 📌 예시 질문

### ✅ 업무 질문 (Tool 호출)

- "상품권 환불 절차는?"
- "가맹점 등록 취소 사유는?"
- "전통시장법 제26조의6 내용 알려줘"

---

### 💬 일반 대화 (Tool 호출 금지)

- "안녕"
- "오늘 날씨 어때?"
- "점심 추천해줘"

---

### 🚀 실행 결과

아래는 실제 실행 시 LangGraph ReAct Agent가  
**일반 대화는 Tool 없이 답변**하고,  
**업무 질문은 반드시 Tool을 호출하여 근거 기반으로 답변**하는 로그 예시입니다.

---

### 💬 일반 대화 예시 (Tool 호출 없음)

사용자가 단순 인사처럼 업무와 무관한 질문을 하면  
Agent는 Tool을 호출하지 않고 자연스럽게 응답합니다.

```bash
[1] HumanMessage: 안녕
--- [Agent] Decide Step ---

[DEBUG] ===== Graph Result Messages =====
[2] AIMessage: 안녕하세요 😊  
오늘 기분은 어떠세요?

📌 ANSWER SOURCE CHECK
💬 이번 답변은 Tool 호출을 사용하지 않은 답변입니다.
```

---

### 🔍 업무 질문 예시 1 (법령 Tool 호출)

업무/법령 질문이 들어오면 Agent는 자동으로 법령 Tool을 호출하여
조항 근거 기반 답변을 생성합니다.

```bash
[3] HumanMessage: 가맹점 등록 취소 사유는 법적으로 무엇이 있나요?
--- [Agent] Decide Step ---

[5] ToolMessage:
{"tool": "search_market_law", ...}

[6] AIMessage:
가맹점 등록 취소 사유로는 거짓 등록이나 등록 제한업종을 영위하는 경우가 있습니다.

[출처]
- 전통시장법 제26조의6

📌 ANSWER SOURCE CHECK
✅ 이번 답변은 Tool 호출 기반입니다.
사용 Tool: search_market_law
```

---

### 🔍 업무 질문 예시 2 (상품권 매뉴얼 Tool 호출)

상품권 유효기간과 같은 업무 질문은
온누리상품권 매뉴얼 문서를 검색하여 답변합니다.

```bash
[7] HumanMessage: 지류 온누리상품권 유효기간이 만료됐더라도 사용이 가능한가요?
--- [Agent] Decide Step ---

[9] ToolMessage:
{"tool": "search_gift", ...}

[10] AIMessage:
지류 온누리상품권의 유효기간은 발행 연도로부터 5년이며,
만료된 상품권은 사용이 불가능합니다.

[출처]
- 온누리상품권_사용자지침서.pdf

📌 ANSWER SOURCE CHECK
✅ 이번 답변은 Tool 호출 기반입니다.
사용 Tool: search_gift
```

---

### 🔍 업무 질문 예시 3 (가맹점 혜택 검색)

가맹점 혜택 문의 문서 기반으로 검색 후 답변합니다.

```bash
[11] HumanMessage: 가맹점을 신청했을 때 혜택이 있나요?
--- [Agent] Decide Step ---

[13] ToolMessage:
{"tool": "search_gift", ...}

[14] AIMessage:
온누리상품권 가맹점으로 등록하시면 고객 유입 효과를 얻을 수 있으며,
환전 및 카드 수수료가 없습니다.

또한, 가맹점 스티커, 안내 리플릿, 포스터 등 홍보물이 무상 제공됩니다.

[출처]
- 온누리상품권_사용자지침서.pdf

📌 ANSWER SOURCE CHECK
✅ 이번 답변은 Tool 호출 기반입니다.
사용 Tool: search_gift
```

---

### 🛡️ 핵심 동작 요약

일반 대화 → Tool 호출 금지

업무 질문 → Tool 호출 필수

답변은 반드시 문서/법령 근거 포함

사용자별 세션은 SQLite Checkpoint로 유지됨

