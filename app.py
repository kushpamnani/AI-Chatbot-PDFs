import streamlit as st
import os
from pypdf import PdfReader
from dotenv import load_dotenv
import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI CHATBOT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load external CSS file for better code organization"""
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

# Header
st.markdown("# AI Document Chatbot")
st.markdown("")

# ---------- Initialize session state (MUST be before using it) ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

if "current_doc" not in st.session_state:
    st.session_state.current_doc = None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 1. UPLOAD DOCUMENTS")
    st.markdown("")
    
    upload_option = st.radio(
        "Choose document source:",
        ["Use Sample Document", "Upload My Own PDF"],
        label_visibility="collapsed"
    )
    
    uploaded_file = None
    use_sample = (upload_option == "Use Sample Document")
    
    if not use_sample:
        st.markdown("")
        uploaded_file = st.file_uploader(
            "Click to upload or drag & drop",
            type="pdf",
            help="PDF files only (max 200MB)",
            accept_multiple_files=False,
            label_visibility="visible"
        )
        
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name}")
            st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")
    else:
        if os.path.exists("data/ai-research-paper.pdf"):
            st.success("✅ Sample document loaded")
        else:
            st.warning("⚠️ Sample not found")
    
    st.markdown("---")
    st.markdown("### 2. CHAT WITH YOUR DOCUMENTS")
    st.caption("Ask questions about your PDF")
    
    # New Chat button (only when a doc is loaded)
    if st.session_state.rag_initialized:
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ---------- RAG setup ----------
@st.cache_resource(show_spinner=False)
def setup_rag_chain(_pdf_file=None, pdf_path=None, original_filename=None):
    """Sets up the complete RAG pipeline"""
    
    if _pdf_file is not None:
        reader = PdfReader(_pdf_file)
        # use the uploaded filename, not the temp file path
        doc_name = original_filename if original_filename else "Uploaded document"
    elif pdf_path is not None:
        if not os.path.exists(pdf_path):
            return None, None, "PDF not found"
        reader = PdfReader(pdf_path)
        doc_name = os.path.basename(pdf_path)
    else:
        return None, None, "No PDF provided"
    
    # Extract text
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    
    if not text.strip():
        return None, None, "Could not extract text from PDF"
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Vector store
    collection_name = f"docs_{hash(doc_name)}"
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="chroma_db"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # LLM
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None, None, "GROQ_API_KEY not found"
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_api_key,
        max_tokens=500
    )
    
    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful AI assistant. Use ONLY the following context to answer.

If the answer is not in the context, say: "I cannot answer this question, please try with a different question"

Context:
{context}

Question: {question}

Answer:"""
    )
    
    # RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, len(chunks), doc_name

# ---------- Process document ----------
rag_chain = None
num_chunks = 0
doc_name = None

with st.spinner("⏳ Processing document..."):
    if use_sample:
        rag_chain, num_chunks, doc_name = setup_rag_chain(
            pdf_path="data/ai-research-paper.pdf"
        )
        if rag_chain:
            st.session_state.rag_initialized = True
            st.session_state.current_doc = "sample"
        else:
            st.error("❌ Could not load sample document")
            st.session_state.rag_initialized = False
    else:
        if uploaded_file is not None:
            original_filename = uploaded_file.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            with open(tmp_path, "rb") as f:
                rag_chain, num_chunks, doc_name = setup_rag_chain(
                    _pdf_file=f,
                    original_filename=original_filename
                )
            
            os.unlink(tmp_path)
            
            if rag_chain:
                st.session_state.rag_initialized = True
                new_doc = uploaded_file.name
                if st.session_state.current_doc != new_doc:
                    st.session_state.messages = []
                    st.session_state.current_doc = new_doc
            else:
                st.error("❌ Could not process PDF")
                st.session_state.rag_initialized = False
        else:
            st.info("👈 Upload a PDF document to start chatting")
            st.session_state.rag_initialized = False

# ---------- Document info ----------
if st.session_state.rag_initialized and rag_chain:
    st.markdown(
        f"""
        <div class="doc-info">
            <p><strong>📄 Document:</strong> {doc_name}</code></p>
            <p><strong>🔢 Chunks Indexed:</strong> {num_chunks}</code></p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Chat history ----------
if st.session_state.rag_initialized and rag_chain:
    for message in st.session_state.messages:
        if message["role"] == "user":
            # User message - right aligned, green
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin: 1.5rem 0;">
                <div style="
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white;
                    padding: 1rem 1.5rem;
                    border-radius: 18px;
                    border-bottom-right-radius: 4px;
                    max-width: 70%;
                    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
                    font-size: 0.95rem;
                    font-weight: 500;
                ">
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # AI message - left aligned, dark gray
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-start; margin: 1.5rem 0;">
                <div style="
                    background-color: #1e293b;
                    color: #e8e8e8;
                    padding: 1.25rem 1.75rem;
                    border-radius: 18px;
                    border-bottom-left-radius: 4px;
                    max-width: 85%;
                    border: 1px solid #334155;
                    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
                    font-size: 1rem;
                    line-height: 1.7;
                ">
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------- Chat input ----------
if st.session_state.rag_initialized and rag_chain:
    if user_input := st.chat_input("Ask something about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message immediately
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; margin: 1.5rem 0;">
            <div style="
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 18px;
                border-bottom-right-radius: 4px;
                max-width: 70%;
                box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
                font-size: 0.95rem;
                font-weight: 500;
            ">
                {user_input}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show loading indicator
        with st.spinner("🔍 Searching..."):
            try:
                response = rag_chain.invoke(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display AI response immediately
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 1.5rem 0;">
                    <div style="
                        background-color: #1e293b;
                        color: #e8e8e8;
                        padding: 1.25rem 1.75rem;
                        border-radius: 18px;
                        border-bottom-left-radius: 4px;
                        max-width: 85%;
                        border: 1px solid #334155;
                        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
                        font-size: 1rem;
                        line-height: 1.7;
                    ">
                        {response}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)
        
        # Rerun to refresh the display
        st.rerun()
else:
    st.chat_input("Upload a document first...", disabled=True)
