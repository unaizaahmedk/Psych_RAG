import gradio as gr
from app.helper import prepare_qa_chain


qa_chain = prepare_qa_chain()


def generate_answer_gradio(message, history=None):
    response = qa_chain.invoke({"query": message})
    answer_text = response['result'].split('Answer:')[-1].strip()
    return answer_text


psych_chatbot = gr.ChatInterface(
    generate_answer_gradio,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(
        placeholder="Ask me any question related to Psychology!",
        container=False,
        scale=7,
    ),
    title="PsychChatMistral",
    theme="soft",
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn=None,
    submit_btn="Submit",
)


if __name__ == "__main__":
    psych_chatbot.launch(debug=True)
