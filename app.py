import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.memory import ConversationBufferMemory
from ingestion import run_ingestion
from llm import create_conversational_chain, summarize_document

# Constants
DOCS_DIR = "docs"
DB_PATH = "chroma_db"

def save_uploaded_files(uploaded_files):
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        
    for uploaded_file in uploaded_files:
        file_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return len(uploaded_files)

# Configure Page
st.set_page_config(page_title="RAG Q&A Bot", page_icon="🤖")

st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>🤖 RAG Neumann Assistant</h1>
        <p style="color: gray; font-size: 1.1em;">Your advanced, fully-local document reasoning assistant powered by SmolLM.</p>
    </div>
    <hr style="opacity: 0.2;">
""", unsafe_allow_html=True)

if len(st.session_state.get("messages", [])) == 0:
    st.info("👋 Welcome! Upload your documents in the sidebar, click 'Process & Ingest', and ask me anything about them!")
# Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
if "chain" not in st.session_state:
    st.session_state.chain = None

@st.cache_resource
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vector_db():
    if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
        return None
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=get_embedding_function()
    )

def setup_chain(token: str = ""):
    vector_db = load_vector_db()
    if vector_db is not None:
        st.session_state.chain = create_conversational_chain(vector_db, st.session_state.memory, token)

# Sidebar
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader("Upload Documents", type=["txt", "pdf", "docx"], accept_multiple_files=True)
    
    if st.button("Process & Ingest"):
        has_existing_docs = os.path.exists(DOCS_DIR) and len(os.listdir(DOCS_DIR)) > 0
        if not uploaded_files and not has_existing_docs:
            st.warning("Please upload files first.")
        else:
            if uploaded_files:
                with st.spinner("Saving documents..."):
                    num_saved = save_uploaded_files(uploaded_files)
                    st.success(f"Saved {num_saved} document(s).")
            
            
            with st.spinner("Ingesting into Vector DB..."):
                try:
                    run_ingestion(doc_path=DOCS_DIR)
                    st.success("Ingestion complete!")
                    setup_chain(os.environ.get("HF_TOKEN"))
                    st.rerun()  # Refresh so state gets correctly updated across board
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")

    st.divider()
    st.header("Database Tools")
    st.page_link("pages/visualize_chroma.py", label="Visualize Chroma DB", icon="📊")

    st.divider()
    st.header("Summarize Document")
    if os.path.exists(DOCS_DIR) and os.listdir(DOCS_DIR):
        files = [f for f in os.listdir(DOCS_DIR) if f.endswith(('.txt', '.pdf', '.docx'))]
        if files:
            selected_file = st.selectbox("Select file to summarize:", files)
            if st.button("Summarize"):
                with st.spinner("Summarizing..."):
                    try:
                        file_path = os.path.join(DOCS_DIR, selected_file)
                        # Abstract document handling
                        from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
                        if selected_file.endswith(".pdf"):
                            loader = PyPDFLoader(file_path)
                        elif selected_file.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        else:
                            loader = TextLoader(file_path)
                        
                        docs = loader.load()
                        full_text = "\n".join([d.page_content for d in docs])
                        
                        summary = summarize_document(full_text, token=os.environ.get("HF_TOKEN"))
                        
                        st.success("Summary Generated!")
                        with st.expander("View Summary", expanded=True):
                            st.write(summary)
                    except Exception as e:
                        st.error(f"Failed to summarize: {e}")
        else:
            st.info("Upload documents to use summarization.")
    else:
        st.info("Upload documents to use summarization.")

# Main Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for src in message["sources"]:
                    st.write(f"- {src}")

if query := st.chat_input("Ask a question about your documents..."):
    if st.session_state.chain is None:
        st.error("Vector database is empty. Please upload and ingest documents first.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chain({"question": query})
                
                answer = response["answer"]
                source_docs = response.get("source_documents", [])
                
                # Format sources
                source_names = list(set([doc.metadata.get('source', 'Unknown') for doc in source_docs]))
                
                st.markdown(answer)
                
                if source_names:
                    with st.expander("View Sources"):
                        for src in source_names:
                            st.write(f"- {src}")
                            
        # Save to state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": source_names
        })

# Initialize the chain lazily after the UI has fully rendered to prevent blank screens
current_token = os.environ.get("HF_TOKEN")
if st.session_state.chain is None or st.session_state.get("_last_token") != current_token:
    if current_token:
        with st.spinner("Initializing AI Engine (This might take a minute if downloading weights for the first time)..."):
            setup_chain(current_token)
            st.session_state._last_token = current_token
    elif os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        st.sidebar.warning("Please provide a Hugging Face Token to initialize the AI Engine.")
