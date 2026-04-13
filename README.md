# AI-Powered Personal Knowledge Assistant 🤖

An advanced, AI-powered personal knowledge assistant that allows users to upload documents (PDFs, DOCX, TXT) and query them using natural language. This project replaces generic web searches by acting as your personalized local knowledge repository. Utilizing Retrieval-Augmented Generation (RAG), the system performs semantic search, retrieves mathematically relevant information, and generates accurate, context-aware answers with explicit source citations.

## ✨ Features
- **Multi-Format Document Support**: Seamlessly upload and process `.txt`, `.pdf`, and `.docx` files directly from a sleek Streamlit sidebar.
- **Smart Chunking & Deterministic Ingestion**: Dynamically chunks large documents and utilizes deterministic IDs to avoid data duplication in the vector database across multiple sessions.
- **Semantic Search Engine**: Fast and accurate retrieval of top 'K' contexts utilizing dense embeddings.
- **Document Summarization**: Instantly generate high-level comprehensive summaries of any uploaded document.
- **Conversational Memory**: The LLM retains conversation history, making follow-ups organic and contextual.
- **Source Citations**: Fully transparent answers with direct links to the origin document chunk metadata for easy cross-referencing.

## 🛠️ Architecture & Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Orchestration Framework**: [LangChain](https://www.langchain.com/)
- **Embedding Model**: `all-MiniLM-L6-v2` (via Hugging Face)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (Persistent local storage)
- **Large Language Model (LLM)**: `meta-llama/Llama-3.1-8B-Instruct:novita` (via Hugging Face Inference API)

### Under the Hood
1. **Ingestion Pipeline (`ingestion.py`)**: Uses LangChain's Document Loaders to extract text. The documents are split into discrete overlapping chunks (Size: 500, Overlap: 50) designed to bypass context limits without losing contextual meaning.
2. **Embedding & Storage (`ingestion.py`)**: Texts are transformed into multidimensional vectors locally using Hugging Face embeddings and persistently stored within a Chroma database.
3. **Retrieval & Generation (`llm.py` & `app.py`)**: A `ConversationalRetrievalChain` captures user queries, finds the closest contexts in Chroma, and injects them into a strict `PromptTemplate`. The Llama 3.1 8B model then generates a well-reasoned response securely via API. 

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- A Hugging Face account and an Access Token (to access Llama 3.1 8B capabilities).

### Installation

1. **Clone the repository** (or navigate to the project directory):
   ```bash
   cd AI-Powered-Personal-Knowledge-Assistant---FOAI-CAPSTONE-PROJECT
   ```

2. **Create and Activate a Virtual Environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure you have all the required libraries installed:
   ```bash
   pip install streamlit langchain langchain-chroma langchain-huggingface langchain-community huggingface_hub pypdf docx2txt chromadb
   ```

4. **Set your Hugging Face Token**:
   Since the app securely interfaces with large LLaMA models, export your Hugging Face API key as an environment variable:
   ```bash
   export HF_TOKEN="your_hugging_face_token_here"
   # For Windows Command Prompt: set HF_TOKEN="your_hugging_face_token_here"
   # For Windows PowerShell: $env:HF_TOKEN="your_hugging_face_token_here"
   ```

### Running the App
1. Execute the Streamlit run command:
   ```bash
   streamlit run app.py
   ```
2. The application UI will launch locally at `http://localhost:8501`.
3. Upload documents via the sidebar workspace and click **Process & Ingest**.
4. Ask questions directly in the chat input or use the Document Summarization drop-down tool!

## 📁 Project Structure

- `app.py`: Main entry point containing the Streamlit App interface and configuration.
- `ingestion.py`: Handles file extraction, systematic chunking, embeddings setup, and Chroma VectorStore initialization.
- `llm.py`: Configures the custom online Hugging Face LLM integration, conversational memory architecture, LangChain prompt orchestration, and document summarization pipelines.
- `docs/`: Default staging directory for user-uploaded documents (generated on runtime).
- `chroma_db/`: Persistent directory housing vector embeddings to ensure fast consecutive launches.

---
*Developed as a Capstone Project exploring the power of Context-Aware AI Systems.*
