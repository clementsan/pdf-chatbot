---
title: PDF Chatbot
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.16.1
app_file: app.py
pinned: true
---


[![](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)



**Aim: PDF-based AI chatbot with retrieval augmented generation**


**Architecture / Tech stack:**
 - Front-end: 
   - user interface via Gradio library
 - Back-end: 
   - HuggingFace embeddings
   - HuggingFace Inference API for open-source LLMs
   - Chromadb vector database
   - LangChain conversational retrieval chain


You can try out the deployed [Hugging Face Space](https://huggingface.co/spaces/cvachet/pdf-chatbot)!


----

### Overview

**Description:**
This AI assistant, using Langchain and open-source LLMs, performs retrieval-augmented generation (RAG) from your PDF documents. The user interface explicitely shows multiple steps to help understand the RAG workflow. This chatbot takes past questions into account when generating answers (via conversational memory), and includes document references for clarity purposes. It leverages small LLM models to run directly on CPU hardware. 


**Available open-source LLMs:**
 - Meta Llama series
 - Alibaba Qwen2.5 series
 - Mistral AI models
 - Microsoft Phi-3.5 series
 - Google Gemma models
 - HuggingFace zephyr and SmolLM series


### Local execution

Command line for execution:
> python3 app.py

The Gradio web application should now be accessible at http://localhost:7860

