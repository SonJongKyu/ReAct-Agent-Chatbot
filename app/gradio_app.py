import gradio as gr
from app.main import run_rag_agent


# ===============================
# ChatGPT 스타일 중앙정렬 CSS
# ===============================
CUSTOM_CSS = """
footer {display:none !important;}
#api-info {display:none !important;}

body {
    background: #f7f7f8 !important;
}

/* ===============================
   전체 레이아웃 중앙정렬
=============================== */
.gradio-container {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}

/* ===============================
   Header 중앙
=============================== */
#header {
    width: 760px !important;
    text-align: center !important;
    margin: 8px auto 10px auto !important;
}

/* ===============================
   Chatbox 중앙
=============================== */
#chatbox {
    width: 760px !important;
    height: calc(100vh - 280px) !important;

    background: white !important;
    border-radius: 18px !important;
    padding: 20px !important;

    border: 1px solid rgba(0,0,0,0.12) !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;

    margin: 0 auto !important;
}

/* ===============================
   입력창 wrapper
=============================== */
#composer_wrap {
    width: 760px !important;
    margin: 15px auto 0 auto !important;

    position: sticky !important;
    bottom: 20px !important;

    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;

    background: transparent !important;
}

/* Gradio block auto-margin 제거 */
#composer_wrap .block {
    margin: 0 auto !important;
}

/* ===============================
   Pill 입력창
=============================== */
#composer {
    width: 100% !important;

    min-height: 54px !important;
    height: auto !important;

    border-radius: 999px !important;

    background: white !important;

    border: 1px solid rgba(0,0,0,0.15) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);

    display: flex !important;
    align-items: center !important;

    padding: 10px 18px !important;
}

/* 여러 줄 입력 시 세로 확장 느낌만 주기 */
#composer:has(textarea:not(:placeholder-shown)) {
    border-radius: 26px !important;
}

/* ===============================
   Textbox 내부 테두리 완전 제거
=============================== */
#msg {
    flex: 1 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

#msg textarea {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;

    background: transparent !important;
    font-size: 15px !important;

    padding: 0 !important;

    resize: none !important;
    max-height: 200px !important;
    overflow-y: auto !important;
}

/* ===============================
   Send 버튼
=============================== */
#send_btn {
    all: unset !important;
    cursor: pointer !important;

    font-size: 22px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    width: 36px !important;
    height: 36px !important;

    margin-left: 8px !important;
}

#send_btn:hover {
    background: rgba(0,0,0,0.06);
    border-radius: 50%;
}

/* ===============================
   안내문구 중앙
=============================== */
#disclaimer {
    width: 100% !important;
    text-align: center !important;

    font-size: 12px !important;
    color: rgba(0,0,0,0.55) !important;

    margin-top: 8px !important;
}

/* ===============================
   typing indicator
=============================== */
.typing {
    display: inline-flex;
    gap: 6px;
}
.typing span {
    width: 8px;
    height: 8px;
    background: #666;
    border-radius: 50%;
    animation: bounce 1.2s infinite ease-in-out;
}
.typing span:nth-child(2){animation-delay:0.2s;}
.typing span:nth-child(3){animation-delay:0.4s;}

@keyframes bounce {
    0%,80%,100% {transform:scale(0);}
    40% {transform:scale(1);}
}

/* ===============================
   로그인 모달 중앙정렬
=============================== */

#popup_overlay {
    position: fixed !important;
    inset: 0 !important;
    background: rgba(0,0,0,0.45) !important;
    z-index: 9998 !important;
}

#popup_box {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;

    width: 420px !important;
    max-width: 90% !important;

    background: white !important;
    border-radius: 18px !important;

    padding: 28px 26px !important;

    box-shadow: 0 10px 40px rgba(0,0,0,0.18) !important;
    z-index: 9999 !important;

    display: flex !important;
    flex-direction: column !important;
    gap: 16px !important;
}

#popup_box.hide,
#popup_box:has(.hidden) {
    display: none !important;
}

#login_id textarea {
    width: 100% !important;
    font-size: 15px !important;
    padding: 12px 14px !important;

    border-radius: 12px !important;
    border: 1px solid rgba(0,0,0,0.18) !important;
}

#login_btn {
    width: 100% !important;
    padding: 12px 0 !important;

    border-radius: 12px !important;
    font-size: 15px !important;
    font-weight: 600 !important;

    background: black !important;
    color: white !important;
    cursor: pointer !important;
}

#login_btn:hover {
    opacity: 0.85 !important;
}
"""


# ===============================
# 사용자별 Respond 함수
# ===============================
def respond(message, history, user_id):

    message = (message or "").strip()
    if not message:
        yield history, ""
        return

    if history is None:
        history = []

    history.append({"role": "user", "content": message})

    loading = """
    <div class="typing">
        <span></span><span></span><span></span>
    </div>
    """
    history.append({"role": "assistant", "content": loading})

    yield history, ""

    answer = run_rag_agent(message, user_id=user_id)

    history[-1] = {"role": "assistant", "content": answer}

    yield history, ""


# ===============================
# 로그인 처리 함수
# ===============================
def start_chat(user_id):

    user_id = (user_id or "").strip()

    if not user_id:
        return (
            gr.update(visible=True),   # login_screen 유지
            gr.update(visible=True),   # overlay 유지
            gr.update(visible=False),  # chat_screen 숨김
            ""
        )

    print(f"[LOGIN] User ID set: {user_id}")

    return (
        gr.update(visible=False),  # login_popup 숨김
        gr.update(visible=False),  # overlay 숨김
        gr.update(visible=True),   # chat_screen 표시
        user_id
    )


# ===============================
# UI 구성
# ===============================
with gr.Blocks() as demo:

    user_state = gr.State("")

    # ===============================
    # 로그인 모달 (최초 화면)
    # ===============================
    overlay = gr.HTML("<div id='popup_overlay'></div>", visible=True)

    with gr.Column(visible=True, elem_id="popup_box") as login_screen:

        gr.Markdown("## 🔑 사용자 ID 입력")
        gr.Markdown("채팅을 시작하려면 ID를 입력하세요")

        user_id_box = gr.Textbox(
            placeholder="예: 담당자A / 고객001",
            show_label=False,
            elem_id="login_id"
        )

        start_btn = gr.Button("채팅 시작", elem_id="login_btn")

    # ===============================
    # 채팅 화면 (로그인 후 활성화)
    # ===============================
    with gr.Column(visible=False, elem_id="main_chat_ui") as chat_screen:

        with gr.Column(elem_id="header"):
            gr.Markdown(
                """
# 🤖 온누리상품권 업무 지원 상담 챗봇 🤖  
업무 담당자용 테스트 환경입니다.
"""
            )

        chatbot = gr.Chatbot(elem_id="chatbox")

        with gr.Column(elem_id="composer_wrap"):

            with gr.Row(elem_id="composer"):

                msg = gr.Textbox(
                    elem_id="msg",
                    placeholder="무엇이든 물어보세요",
                    lines=1,
                    show_label=False,
                    container=False,
                )

                send = gr.Button("➤", elem_id="send_btn")

            gr.Markdown(
                "LLM은 실수를 할 수 있습니다. 중요한 정보는 확인이 필요합니다.",
                elem_id="disclaimer"
            )

        send.click(
            respond,
            inputs=[msg, chatbot, user_state],
            outputs=[chatbot, msg]
        )

        msg.submit(
            respond,
            inputs=[msg, chatbot, user_state],
            outputs=[chatbot, msg]
        )

    # ===============================
    # 로그인 버튼 이벤트
    # ===============================
    start_btn.click(
        start_chat,
        inputs=[user_id_box],
        outputs=[login_screen, overlay, chat_screen, user_state]
    )

    user_id_box.submit(
        start_chat,
        inputs=[user_id_box],
        outputs=[login_screen, overlay, chat_screen, user_state]
    )


# ===============================
# 실행
# ===============================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        css=CUSTOM_CSS,
    )
