import gradio as gr


def mock_chat(message, history):
    """
    这里只做 UI 演示，不执行 SQL，不连接数据库。
    """
    history = history or []

    history.append({
        "role": "user",
        "content": message
    })

    history.append({
        "role": "assistant",
        "content": "这里是 ChatBot 回复区域。当前版本只包含 UI，没有执行 SQL 查询。"
    })

    return history, ""


with gr.Blocks(title="IRIS Health Care Chat UI") as demo:
    gr.Markdown("## IRIS Health Care Chat UI")
    gr.Markdown("上方为结果展示区域，下方为 ChatBot 输入区域。")

    # 上半部分：结果展示区
    with gr.Group():
        gr.Markdown("### 查询结果 / 数据展示区域")

        result_table = gr.Dataframe(
            headers=["Column 1", "Column 2", "Column 3"],
            value=[],
            label="结果表格",
            interactive=False,
        )

    # 下半部分：ChatBot 区域
    with gr.Group():
        gr.Markdown("### ChatBot")

        chatbot = gr.Chatbot(
            label="ChatBot",
        )

        user_input = gr.Textbox(
            label="输入内容",
            placeholder="请输入你的问题...",
            lines=2,
        )

        send_button = gr.Button("发送")

    # 事件绑定：这里只更新聊天窗口，不处理 SQL
    user_input.submit(
        fn=mock_chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input],
    )

    send_button.click(
        fn=mock_chat,
        inputs=[user_input, chatbot],
        outputs=[chatbot, user_input],
    )


demo.launch(server_name="0.0.0.0", server_port=7860)