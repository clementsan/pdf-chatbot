"""
LLM chain retrieval
"""

import json
import gradio as gr

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate


# Add system template for RAG application
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer concise.
Question: {question} 
Context: {context} 
Helpful Answer:
"""


# Initialize langchain LLM chain
def initialize_llmchain(
    llm_model,
    huggingfacehub_api_token,
    temperature,
    max_tokens,
    top_k,
    vector_db,
    progress=gr.Progress(),
):
    """Initialize Langchain LLM chain"""

    progress(0.1, desc="Initializing HF tokenizer...")
    # HuggingFaceHub uses HF inference endpoints
    progress(0.5, desc="Initializing HF Hub...")
    # Use of trust_remote_code as model_kwargs
    # Warning: langchain issue
    # URL: https://github.com/langchain-ai/langchain/issues/6080

    # if 'Llama' in llm_model:
    #     task = "conversational"
    # else:
    #     task = "text-generation"
    # print(f"Task: {task}")

    llm = HuggingFaceEndpoint(
        repo_id=llm_model,
        task="text-generation",
        #task="conversational",
        provider="hf-inference",
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_k=top_k,
        huggingfacehub_api_token=huggingfacehub_api_token,
    )

    progress(0.75, desc="Defining buffer memory...")
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", return_messages=True
    )
    # retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    retriever = vector_db.as_retriever()

    progress(0.8, desc="Defining retrieval chain...")
    with open('prompt_template.json', 'r') as file:
    	system_prompt = json.load(file)
    prompt_template = system_prompt["prompt"]
    rag_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        combine_docs_chain_kwargs={"prompt": rag_prompt},
        return_source_documents=True,
        # return_generated_question=False,
        verbose=False,
    )
    progress(0.9, desc="Done!")

    return qa_chain


def format_chat_history(message, chat_history):
    """Format chat history for llm chain"""

    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history


def invoke_qa_chain(qa_chain, message, history):
    """Invoke question-answering chain"""

    formatted_chat_history = format_chat_history(message, history)
    # print("formatted_chat_history",formatted_chat_history)

    # Generate response using QA chain
    response = qa_chain.invoke(
        {"question": message, "chat_history": formatted_chat_history}
    )

    response_sources = response["source_documents"]

    response_answer = response["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]

    # Append user message and response to chat history
    new_history = history + [(message, response_answer)]

    # print ('chat response: ', response_answer)
    # print('DB source', response_sources)

    return qa_chain, new_history, response_sources
