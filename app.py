import gradio as gr
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

from pathlib import Path
import chromadb
from unidecode import unidecode

from transformers import AutoTokenizer
import transformers
import torch
import tqdm 
import accelerate
import re

from dotenv import load_dotenv


# Load environment file - HuggingFace API key
_ = load_dotenv()
huggingfacehub_api_token = os.environ.get("HUGGINGFACE_API_KEY")


# Add system template for RAG application
prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer concise.
Question: {question} 
Context: {context} 
Helpful Answer:
""" 


# default_persist_directory = './chroma_HF/'
list_llm = ["mistralai/Mistral-7B-Instruct-v0.3", "microsoft/Phi-3.5-mini-instruct", \
    "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct", \
    "HuggingFaceH4/zephyr-7b-beta", "HuggingFaceH4/zephyr-7b-gemma-v0.1", \
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "google/gemma-2-2b-it", "google/gemma-2-9b-it", \
    "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]


# Load PDF document and create doc splits
def load_doc(list_file_path, chunk_size, chunk_overlap):
    """Load PDF document and create doc splits"""

    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits


# Create vector database
def create_db(splits, collection_name):
    """Create embeddings and vector database"""

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        # model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        # encode_kwargs={'normalize_embeddings': False}
    )
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        client=new_client,
        collection_name=collection_name,
        # persist_directory=default_persist_directory
    )
    return vectordb



# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    """Initialize Langchain LLM chain"""

    progress(0.1, desc="Initializing HF tokenizer...")
    # HuggingFaceHub uses HF inference endpoints
    progress(0.5, desc="Initializing HF Hub...")
    # Use of trust_remote_code as model_kwargs
    # Warning: langchain issue
    # URL: https://github.com/langchain-ai/langchain/issues/6080

    
    llm = HuggingFaceEndpoint(
        repo_id=llm_model,
        task = "text-generation",
        temperature = temperature,
        max_new_tokens = max_tokens,
        top_k = top_k,
        huggingfacehub_api_token=huggingfacehub_api_token,
    )
    
    progress(0.75, desc="Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    # retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    retriever=vector_db.as_retriever()
    progress(0.8, desc="Defining retrieval chain...")
    rag_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"]) 
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        combine_docs_chain_kwargs={"prompt": rag_prompt},
        return_source_documents=True,
        #return_generated_question=False,
        verbose=False,
    )
    progress(0.9, desc="Done!")

    return qa_chain


# Generate collection name for vector database
#  - Use filepath as input, ensuring unicode text
def create_collection_name(filepath):
    # Extract filename without extension
    collection_name = Path(filepath).stem
    # Fix potential issues from naming convention
    ## Remove space
    collection_name = collection_name.replace(" ","-") 
    ## ASCII transliterations of Unicode text
    collection_name = unidecode(collection_name)
    ## Remove special characters
    #collection_name = re.findall("[\dA-Za-z]*", collection_name)[0]
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    ## Limit length to 50 characters
    collection_name = collection_name[:50]
    ## Minimum length of 3 characters
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    ## Enforce start and end as alphanumeric character
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    print('\n\nFilepath: ', filepath)
    print('Collection name: ', collection_name)
    return collection_name


# Initialize database
def initialize_database(list_file_obj, chunk_size, chunk_overlap, progress=gr.Progress()):
    # Create list of documents (when valid)
    list_file_path = [x.name for x in list_file_obj if x is not None]
    # Create collection_name for vector database
    progress(0.1, desc="Creating collection name...")
    collection_name = create_collection_name(list_file_path[0])
    progress(0.25, desc="Loading document...")
    # Load document and create splits
    doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
    # Create or load vector database
    progress(0.5, desc="Generating vector database...")
    # global vector_db
    vector_db = create_db(doc_splits, collection_name)
    progress(0.9, desc="Done!")
    return vector_db, collection_name, "Complete!"


def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    # print("llm_option",llm_option)
    llm_name = list_llm[llm_option]
    print("llm_name: ",llm_name)
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "Complete!"


def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history
    

def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    #print("formatted_chat_history",formatted_chat_history)
   
    # Generate response using QA chain
    response = qa_chain.invoke({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    # Langchain sources are zero-based
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    # print ('chat response: ', response_answer)
    # print('DB source', response_sources)
    
    # Append user message and response to chat history
    new_history = history + [(message, response_answer)]
    # return gr.update(value=""), new_history, response_sources[0], response_sources[1] 
    return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page
    

def demo():
    with gr.Blocks(theme="base") as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        collection_name = gr.State()
        
        gr.Markdown(
        """<center><h2>PDF-based chatbot</center></h2>
        <h3>Ask any questions about your PDF documents</h3>""")
        gr.Markdown(
        """<b>Note:</b> This AI assistant, using Langchain and open-source LLMs, performs retrieval-augmented generation (RAG) from your PDF documents. \
        The user interface explicitely shows multiple steps to help understand the RAG workflow. 
        This chatbot takes past questions into account when generating answers (via conversational memory), and includes document references for clarity purposes.<br>
        <br><b>Warning:</b> This space uses the free CPU Basic hardware from Hugging Face. Some steps and LLM models used below (free inference endpoints) can take some time to generate a reply.
        """)
        
        with gr.Tab("Step 1 - Upload PDF"):
            with gr.Row():
                document = gr.File(height=200, file_count="multiple", file_types=[".pdf"], interactive=True, label="Upload your PDF documents (single or multiple)")
        
        with gr.Tab("Step 2 - Process document"):
            with gr.Row():
                db_btn = gr.Radio(["ChromaDB"], label="Vector database type", value = "ChromaDB", type="index", info="Choose your vector database")
            with gr.Accordion("Advanced options - Document text splitter", open=False):
                with gr.Row():
                    slider_chunk_size = gr.Slider(minimum = 100, maximum = 1000, value=600, step=20, label="Chunk size", info="Chunk size", interactive=True)
                with gr.Row():
                    slider_chunk_overlap = gr.Slider(minimum = 10, maximum = 200, value=40, step=10, label="Chunk overlap", info="Chunk overlap", interactive=True)
            with gr.Row():
                db_progress = gr.Textbox(label="Vector database initialization", value="None")
            with gr.Row():
                db_btn = gr.Button("Generate vector database")
            
        with gr.Tab("Step 3 - Initialize QA chain"):
            with gr.Row():
                llm_btn = gr.Radio(list_llm_simple, \
                    label="LLM models", value = list_llm_simple[0], type="index", info="Choose your LLM model")
            with gr.Accordion("Advanced options - LLM model", open=False):
                with gr.Row():
                    slider_temperature = gr.Slider(minimum = 0.01, maximum = 1.0, value=0.7, step=0.1, label="Temperature", info="Model temperature", interactive=True)
                with gr.Row():
                    slider_maxtokens = gr.Slider(minimum = 224, maximum = 4096, value=1024, step=32, label="Max Tokens", info="Model max tokens", interactive=True)
                with gr.Row():
                    slider_topk = gr.Slider(minimum = 1, maximum = 10, value=3, step=1, label="top-k samples", info="Model top-k samples", interactive=True)
            with gr.Row():
                llm_progress = gr.Textbox(value="None",label="QA chain initialization")
            with gr.Row():
                qachain_btn = gr.Button("Initialize Question Answering chain")

        with gr.Tab("Step 4 - Chatbot"):
            chatbot = gr.Chatbot(height=300)
            with gr.Accordion("Advanced - Document references", open=False):
                with gr.Row():
                    doc_source1 = gr.Textbox(label="Reference 1", lines=2, container=True, scale=20)
                    source1_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source2 = gr.Textbox(label="Reference 2", lines=2, container=True, scale=20)
                    source2_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    doc_source3 = gr.Textbox(label="Reference 3", lines=2, container=True, scale=20)
                    source3_page = gr.Number(label="Page", scale=1)
            with gr.Row():
                msg = gr.Textbox(placeholder="Type message (e.g. 'Can you summarize this document in one paragraph?')", container=True)
            with gr.Row():
                submit_btn = gr.Button("Submit message")
                clear_btn = gr.ClearButton(components=[msg, chatbot], value="Clear conversation")
            
        # Preprocessing events
        db_btn.click(initialize_database, \
            inputs=[document, slider_chunk_size, slider_chunk_overlap], \
            outputs=[vector_db, collection_name, db_progress])
        qachain_btn.click(initialize_LLM, \
            inputs=[llm_btn, slider_temperature, slider_maxtokens, slider_topk, vector_db], \
            outputs=[qa_chain, llm_progress]).then(lambda:[None,"",0,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)

        # Chatbot events
        msg.submit(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
        submit_btn.click(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
        clear_btn.click(lambda:[None,"",0,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
    demo.queue().launch(debug=True)


if __name__ == "__main__":
    demo()
