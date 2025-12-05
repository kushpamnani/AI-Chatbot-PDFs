# 🤖 AI Document Chatbot

An AI-powered document question answering app that lets you upload any PDF and chat with it using Retrieval-Augmented Generation (RAG). Built with Streamlit, LangChain, ChromaDB, HuggingFace embeddings, and Groq’s LLM.

---

## 🚀 Features

- 📁 Upload your own PDF or use a sample document
- 💬 Ask natural-language questions about the document
- 🧠 RAG pipeline: semantic retrieval + LLM generation
- 🔍 Chunked document indexing for efficient search
- 🧾 Clean chat-style interface with distinct user/AI bubbles
- 📄 Displays current document name and number of indexed chunks

---

## 🧱 Tech Stack

- **Frontend / UI:** Streamlit
- **Orchestration:** LangChain
- **Vector Store:** ChromaDB
- **Embeddings:** HuggingFace **`all-MiniLM-L6-v2`**
- **LLM:** Groq **`llama-3.1-8b-instant`** via **`langchain_groq`**
- **PDF Parsing:** **`pypdf`**
- **Config:** **`python-dotenv`**

---

## 📂 Project Structure

`pdf-qa-chatbot/
├── app.py               *# Main Streamlit app*
├── requirements.txt     *# Python dependencies*
├── .env                 *# Environment variables (not committed)*
├── assets/
│   └── style.css        *# Custom dark theme + chat styling*
├── data/
│   └── ai-research-paper.pdf   *# Sample PDF*
└── chroma_db/           *# Chroma vector store (auto-created)*`

---

## 🔐 Environment Variables

Create a **`.env`** file in the project root:

`GROQ_API_KEY=your_groq_api_key_here`

You can get an API key from the Groq console.

---

## ⚙️ Setup & Installation

1. **Clone the repo:**
    
    `git clone https://github.com/your-username/ai-document-chatbot.git
    cd ai-document-chatbot`
    
2. **Create and activate virtual environment:**
    
    `python -m venv .venv
    source .venv/bin/activate    *# macOS/Linux# .venv\Scripts\activate     # Windows*`
    
3. **Install dependencies:**
    
    `pip install --upgrade pip
    pip install -r requirements.txt`
    
4. **Add `.env` with your `GROQ_API_KEY`** (see above).
5. **Run the app:**
    
    `python -m streamlit run app.py`
    
6. Open in your browser:
    
    **`http://localhost:8501`**
    

---


## 🧪 Typical Usage

1. Start the app.
2. In the sidebar:
    - Choose **“Use Sample Document”** or **“Upload My Own PDF”**.
3. Wait for the “Processing document…” spinner to finish.
4. Check the document info card:
    - Document name
    - Number of chunks indexed
5. Use the chat box at the bottom to ask questions like:
    - “Summarize this document.”
    - “What are the main findings?”
    - “Who is the intended audience?”

---

## 🧹 Notes & Limitations

- Answers are limited to the content in the uploaded PDF.
- If relevant context is not found, the bot responds that it cannot answer the question.
- First question after uploading a large PDF may take a few seconds while embeddings and Chroma index are built.