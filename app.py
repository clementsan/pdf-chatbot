import gradio as gr
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub

from transformers import AutoTokenizer
import transformers
import torch
import tqdm 
import accelerate


default_persist_directory = './chroma_HF/'

llm_name1 = "mistralai/Mistral-7B-Instruct-v0.2"
llm_name2 = "mistralai/Mistral-7B-Instruct-v0.1"
llm_name3 = "meta-llama/Llama-2-7b-chat-hf"
llm_name4 = "microsoft/phi-2"
llm_name5 = "mosaicml/mpt-7b-instruct"
llm_name6 = "tiiuae/falcon-7b-instruct"
llm_name7 = "google/flan-t5-xxl"
list_llm = [llm_name1, llm_name2, llm_name3, llm_name4, llm_name5, llm_name6, llm_name7]
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

# Load PDF document and create doc splits
def load_doc(list_file_path, chunk_size, chunk_overlap):
    # Processing for one document only
    # loader = PyPDFLoader(file_path)
    # pages = loader.load()
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 50)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits


# Create vector database
def create_db(splits):
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=default_persist_directory
    )
    return vectordb


# Load vector database
def load_db():
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma(
        persist_directory=default_persist_directory, 
        embedding_function=embedding)
    return vectordb


# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    progress(0.1, desc="Initializing HF tokenizer...")
    # HuggingFacePipeline uses local model
    # Warning: it will download model locally...
    # tokenizer=AutoTokenizer.from_pretrained(llm_model)
    # progress(0.5, desc="Initializing HF pipeline...")
    # pipeline=transformers.pipeline(
    #     "text-generation",
    #     model=llm_model,
    #     tokenizer=tokenizer,
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    #     device_map="auto",
    #     # max_length=1024,
    #     max_new_tokens=max_tokens,
    #     do_sample=True,
    #     top_k=top_k,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id
    #     )
    # llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': temperature})
    
    # HuggingFaceHub uses HF inference endpoints
    progress(0.5, desc="Initializing HF Hub...")
    llm = HuggingFaceHub(
        repo_id=llm_model, 
        model_kwargs={"temperature": temperature, "max_new_tokens": max_tokens, "top_k": top_k,\
        "trust_remote_code": True, "torch_dtype": torch.bfloat16}
    )
    
    progress(0.5, desc="Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    # retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    retriever=vector_db.as_retriever()
    progress(0.8, desc="Defining retrieval chain...")
    global qa_chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        # combine_docs_chain_kwargs={"prompt": your_prompt})
        # return_source_documents=True,
        # return_generated_question=True,
        # verbose=True,
    )
    progress(0.9, desc="Done!")
    # return qa_chain


# Initialize all elements
def initialize_database(list_file_obj, chunk_size, chunk_overlap, progress=gr.Progress()):
    # Create list of documents (when valid)
    #file_path = file_obj.name
    list_file_path = [x.name for x in list_file_obj if x is not None]
    print('list_file_path', list_file_path)
    progress(0.25, desc="Loading document...")
    # Load document and create splits
    doc_splits = load_doc(list_file_path, chunk_size, chunk_overlap)
    # Create or load Vector database
    progress(0.5, desc="Generating vector database...")
    # global vector_db
    vector_db = create_db(doc_splits)
    progress(0.9, desc="Done!")
    return vector_db, "Complete!"
    #return qa_chain


def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    print("llm_option",llm_option)
    llm_name = list_llm[llm_option]
    print("llm_name",llm_name)
    initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return "Complete!"
    #return qa_chain


def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history
    

def conversation(message, history):
    formatted_chat_history = format_chat_history(message, history)
    #print("formatted_chat_history",formatted_chat_history)
   
    # Generate response using QA chain
    response = qa_chain({"question": message, "chat_history": formatted_chat_history})
    # return response['answer']
    
    # Append user message and response to chat history
    new_history = history + [(message, response["answer"])]
    return gr.update(value=""), new_history        
    

def upload_file(file_obj):
    list_file_path = []
    for idx, file in enumerate(file_obj):
        file_path = file_obj.name
        list_file_path.append(file_path)
    # print(file_path)
    # initialize_database(file_path, progress)
    return list_file_path


def demo():
    with gr.Blocks(theme="base") as demo:
        vector_db = gr.State()
        # qa_chain = gr.Variable()
        
        gr.Markdown(
        """<center><h2> Document-based chatbot</center></h2>
        <h3>Ask any questions about your PDF documents (single or multiple)</h3>
        <i>Note: chatbot performs question-answering using Langchain and LLMs</i>
        """)
        with gr.Tab("Step 1 - Document pre-processing"):
            with gr.Row():
                document = gr.Files(height=100, file_count="multiple", file_types=["pdf"], interactive=True, label="Upload PDF Documents")
                # upload_btn = gr.UploadButton("Loading document...", height=100, file_count="multiple", file_types=["pdf"], scale=1)
            with gr.Row():
                db_btn = gr.Radio(["ChromaDB"], label="Vector database", value = "ChromaDB", type="index", info="Choose your vector database")
            with gr.Accordion("Advanced options - Text splitter", open=False):
                with gr.Row():
                    slider_chunk_size = gr.Slider(minimum = 100, maximum = 1000, value=600, step=20, label="Chunk size", info="Chunk size", interactive=True)
                with gr.Row():
                    slider_chunk_overlap = gr.Slider(minimum = 10, maximum = 200, value=50, step=10, label="Chunk overlap", info="Chunk overlap", interactive=True)
            with gr.Row():
                db_progress = gr.Textbox(label="Database Initialization", value="None")
            with gr.Row():
                db_btn = gr.Button("Generating vector database...")
            
        with gr.Tab("Step 2 - Initializing QA chain"):
            with gr.Row():
                llm_btn = gr.Radio(list_llm_simple, \
                    label="LLM", value = list_llm_simple[0], type="index", info="Choose your LLM model")
            with gr.Accordion("Advanced options - LLM", open=False):
                slider_temperature = gr.Slider(minimum = 0.0, maximum = 1.0, value=0.7, step=0.1, label="Temperature", info="Model temperature", interactive=True)
                slider_maxtokens = gr.Slider(minimum = 256, maximum = 4096, value=1024, step=24, label="Max Tokens", info="Model max tokens", interactive=True)
                slider_topk = gr.Slider(minimum = 1, maximum = 10, value=3, step=1, label="top-k samples", info="Model top-k samples", interactive=True)
            with gr.Row():
                llm_progress = gr.Textbox(value="None",label="QA chain Initialization")
            with gr.Row():
                qachain_btn = gr.Button("QA chain Initialization...")

        with gr.Tab("Step 3 - Conversation"):
            chatbot = gr.Chatbot(height=300)
            with gr.Row():
                msg = gr.Textbox(placeholder="Type message", container=True)
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.ClearButton([msg, chatbot])
            
        # Preprocessing events
        #upload_btn.upload(upload_file, inputs=[upload_btn], outputs=[document])
        db_btn.click(initialize_database, inputs=[document, slider_chunk_size, slider_chunk_overlap], outputs=[vector_db, db_progress])
        qachain_btn.click(initialize_LLM, inputs=[llm_btn, slider_temperature, slider_maxtokens, slider_topk, vector_db], outputs=[llm_progress]).then(lambda: None, None, chatbot, queue=False)

        # Chatbot events
        msg.submit(conversation, [msg, chatbot], [msg, chatbot], queue=False)
        submit_btn.click(conversation, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False)
        clear_btn.click(lambda: None, None, chatbot, queue=False)
    demo.queue().launch(debug=True)


if __name__ == "__main__":
    demo()
