"""
PDF-based chatbot with Retrieval-Augmented Generation
"""

import os
import gradio as gr

from dotenv import load_dotenv

import indexing
import retrieval


# default_persist_directory = './chroma_HF/'
list_llm = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "HuggingFaceH4/zephyr-7b-gemma-v0.1",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]


# Load environment file - HuggingFace API key
def retrieve_api():
    """Retrieve HuggingFace API Key"""
    _ = load_dotenv()
    global huggingfacehub_api_token
    huggingfacehub_api_token = os.environ.get("HUGGINGFACE_API_KEY")


# Initialize database
def initialize_database(
    list_file_obj, chunk_size, chunk_overlap, progress=gr.Progress()
):
    """Initialize database"""

    # Create list of documents (when valid)
    list_file_path = [x.name for x in list_file_obj if x is not None]

    # Create collection_name for vector database
    progress(0.1, desc="Creating collection name...")
    collection_name = indexing.create_collection_name(list_file_path[0])

    progress(0.25, desc="Loading document...")
    # Load document and create splits
    doc_splits = indexing.load_doc(list_file_path, chunk_size, chunk_overlap)

    # Create or load vector database
    progress(0.5, desc="Generating vector database...")

    # global vector_db
    vector_db = indexing.create_db(doc_splits, collection_name)

    return vector_db, collection_name, "Complete!"


# Initialize LLM
def initialize_llm(
    llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()
):
    """Initialize LLM"""

    # print("llm_option",llm_option)
    llm_name = list_llm[llm_option]
    print("llm_name: ", llm_name)
    qa_chain = retrieval.initialize_llmchain(
        llm_name, huggingfacehub_api_token, llm_temperature, max_tokens, top_k, vector_db, progress
    )
    return qa_chain, "Complete!"


# Chatbot conversation
def conversation(qa_chain, message, history):
    """Chatbot conversation"""

    qa_chain, new_history, response_sources = retrieval.invoke_qa_chain(
        qa_chain, message, history
    )

    # Format output gradio components
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    # Langchain sources are zero-based
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1

    return (
        qa_chain,
        gr.update(value=""),
        new_history,
        response_source1,
        response_source1_page,
        response_source2,
        response_source2_page,
        response_source3,
        response_source3_page,
    )


SPACE_TITLE = """
<center><h2>PDF-based chatbot</center></h2>
<h3>Ask any questions about your PDF documents</h3>
"""

SPACE_INFO = """
<b>Note:</b> This AI assistant, using Langchain and open-source LLMs, performs retrieval-augmented generation (RAG) from your PDF documents. \
The user interface explicitely shows multiple steps to help understand the RAG workflow. 
This chatbot takes past questions into account when generating answers (via conversational memory), and includes document references for clarity purposes.<br>
<br><b>Warning:</b> This space uses the free CPU Basic hardware from Hugging Face. Some steps and LLM models used below (free inference endpoints) can take some time to generate a reply.
"""


# Gradio User Interface
def gradio_ui():
    """Gradio User Interface"""

    with gr.Blocks(theme="base") as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        collection_name = gr.State()

        gr.Markdown(SPACE_TITLE)
        gr.Markdown(SPACE_INFO)

        with gr.Tab("Step 1 - Upload PDF"):
            with gr.Row():
                document = gr.File(
                    height=200,
                    file_count="multiple",
                    file_types=[".pdf"],
                    interactive=True,
                    label="Upload your PDF documents (single or multiple)",
                )

        with gr.Tab("Step 2 - Process document"):
            with gr.Row():
                db_btn = gr.Radio(
                    ["ChromaDB"],
                    label="Vector database type",
                    value="ChromaDB",
                    type="index",
                    info="Choose your vector database",
                )
            with gr.Accordion("Advanced options - Document text splitter", open=False):
                with gr.Row():
                    slider_chunk_size = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=600,
                        step=20,
                        label="Chunk size",
                        info="Chunk size",
                        interactive=True,
                    )
                with gr.Row():
                    slider_chunk_overlap = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=40,
                        step=10,
                        label="Chunk overlap",
                        info="Chunk overlap",
                        interactive=True,
                    )
            with gr.Row():
                db_progress = gr.Textbox(
                    label="Vector database initialization", value="None"
                )
            with gr.Row():
                db_btn = gr.Button("Generate vector database")

        with gr.Tab("Step 3 - Initialize QA chain"):
            with gr.Row():
                llm_btn = gr.Radio(
                    list_llm_simple,
                    label="LLM models",
                    value=list_llm_simple[0],
                    type="index",
                    info="Choose your LLM model",
                )
            with gr.Accordion("Advanced options - LLM model", open=False):
                with gr.Row():
                    slider_temperature = gr.Slider(
                        minimum=0.01,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Model temperature",
                        interactive=True,
                    )
                with gr.Row():
                    slider_maxtokens = gr.Slider(
                        minimum=224,
                        maximum=4096,
                        value=1024,
                        step=32,
                        label="Max Tokens",
                        info="Model max tokens",
                        interactive=True,
                    )
                with gr.Row():
                    slider_topk = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="top-k samples",
                        info="Model top-k samples",
                        interactive=True,
                    )
            with gr.Row():
                llm_progress = gr.Textbox(value="None", label="QA chain initialization")
            with gr.Row():
                qachain_btn = gr.Button("Initialize Question Answering chain")

        with gr.Tab("Step 4 - Chatbot"):
            chatbot = gr.Chatbot(height=300)
            with gr.Accordion("Advanced - Document references", open=False):
                with gr.Row():
                    doc_source1 = gr.Textbox(
                        label="Reference 1", lines=2, container=True, scale=20
                    )
                    source1_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source2 = gr.Textbox(
                        label="Reference 2", lines=2, container=True, scale=20
                    )
                    source2_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source3 = gr.Textbox(
                        label="Reference 3", lines=2, container=True, scale=20
                    )
                    source3_page = gr.Number(label="Page", scale=1)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type message (e.g. 'Can you summarize this document in one paragraph?')",
                    container=True,
                )
            with gr.Row():
                submit_btn = gr.Button("Submit message")
                clear_btn = gr.ClearButton(
                    components=[msg, chatbot], value="Clear conversation"
                )

        # Preprocessing events
        db_btn.click(
            initialize_database,
            inputs=[document, slider_chunk_size, slider_chunk_overlap],
            outputs=[vector_db, collection_name, db_progress],
        )
        qachain_btn.click(
            initialize_llm,
            inputs=[
                llm_btn,
                slider_temperature,
                slider_maxtokens,
                slider_topk,
                vector_db,
            ],
            outputs=[qa_chain, llm_progress],
        ).then(
            lambda: [None, "", 0, "", 0, "", 0],
            inputs=None,
            outputs=[
                chatbot,
                doc_source1,
                source1_page,
                doc_source2,
                source2_page,
                doc_source3,
                source3_page,
            ],
            queue=False,
        )

        # Chatbot events
        msg.submit(
            conversation,
            inputs=[qa_chain, msg, chatbot],
            outputs=[
                qa_chain,
                msg,
                chatbot,
                doc_source1,
                source1_page,
                doc_source2,
                source2_page,
                doc_source3,
                source3_page,
            ],
            queue=False,
        )
        submit_btn.click(
            conversation,
            inputs=[qa_chain, msg, chatbot],
            outputs=[
                qa_chain,
                msg,
                chatbot,
                doc_source1,
                source1_page,
                doc_source2,
                source2_page,
                doc_source3,
                source3_page,
            ],
            queue=False,
        )
        clear_btn.click(
            lambda: [None, "", 0, "", 0, "", 0],
            inputs=None,
            outputs=[
                chatbot,
                doc_source1,
                source1_page,
                doc_source2,
                source2_page,
                doc_source3,
                source3_page,
            ],
            queue=False,
        )
    demo.queue().launch(debug=True)


if __name__ == "__main__":
    retrieve_api()
    gradio_ui()
