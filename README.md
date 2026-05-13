# 💰 Finance RAG Chatbot using LLM and LangChain

An intelligent financial chatbot based on **Retrieval-Augmented Generation (RAG)**, **LangChain**, **Ollama**, and **LLaMA2**.  
The chatbot provides reliable and context-aware financial answers using local and web financial data sources.

---

# 📌 Project Overview

This project was developed as part of a TALN academic project inspired by the paper:

> *“Personalized Finance Chatbot Powered by RAG and Generative AI for Smart Wealth Management”*

The system combines:
- 🔎 Information Retrieval using a vector database
- 🧠 Local Large Language Model (LLaMA2 via Ollama)
- 📚 Financial document processing
- 💬 Intelligent conversational interface with Streamlit

---

# 🚀 Features

- Financial Question Answering
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Local LLM with Ollama (LLaMA2)
- Streamlit Web Interface
- Context-aware Responses
- Financial Data Retrieval
- NLP-based Text Processing

---

# 🛠️ Technologies Used

- Python
- LangChain
- Ollama
- LLaMA2
- Streamlit
- Vector Database (Milvus)
- NLP Techniques
- RAG Architecture

---

# ⚙️ Installation
1️⃣ Create and Activate Environment: 
conda create -n myenv python=3.10

conda activate myenv

2️⃣ Install Dependencies: pip install -r requirements.txt

3️⃣ Install Ollama: 
Download Ollama from: https://ollama.com/download

Then pull the LLaMA2 model: ollama pull llama2

---

# ▶️ Running the Project
⚠️ You need to open two terminals.

🖥️ Terminal 1 — Start LLaMA2: 

conda activate myenv

cd src

ollama run llama2

🖥️ Terminal 2 — Run the Application:

conda activate myenv

cd src

python seed_data.py

streamlit run main.py

---

🎥 Demo Video
Project execution video: https://drive.google.com/drive/u/0/my-drive

---
🧠 Methodology

The chatbot is based on a RAG (Retrieval-Augmented Generation) architecture combining:

Document Retrieval
Semantic Similarity Search
Embedding Generation
Contextual Response Generation
Financial Data Processing

The chatbot only generates responses from retrieved relevant documents to reduce hallucinations and improve reliability.
