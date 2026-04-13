import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil

def load_docs(doc_path = "docs"):
    """Loading Docs here"""
    print(f"Loading docs of {doc_path} directory")
    
    if not os.path.exists(doc_path):
        raise FileNotFoundError("This folder does not exists")

    documents = []
    
    # Load TXT files
    txt_loader = DirectoryLoader(doc_path, glob="**/*.txt", loader_cls=TextLoader)
    documents.extend(txt_loader.load())
    
    # Load PDF files
    pdf_loader = DirectoryLoader(doc_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    # Load DOCX files
    docx_loader = DirectoryLoader(doc_path, glob="**/*.docx", loader_cls=Docx2txtLoader)
    documents.extend(docx_loader.load())
    
    if len(documents) == 0:
        raise ValueError(f"No text, pdf, or docx file exists in {doc_path} folder")
    
    for i,doc in enumerate(documents):
        print(f" --- Document {i+1} --- ")
        print(f" Source: {doc.metadata['source']} ")
        print(f" Length: {len(doc.page_content)} ")
        print(f" Content: {doc.page_content[:200]} ")
        
    return documents


def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    """Create chunks from documents"""
    print(f"Creating chunks with size {chunk_size} and overlap {chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Total chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f" --- Chunk {i+1} --- ")
        print(f" Size: {len(chunk.page_content)} ")
        print(f" Content: {chunk.page_content[:100]} ")
    
    return chunks


def create_embeddings(chunks, model="all-MiniLM-L6-v2"):
    """Create embeddings for chunks using HuggingFace"""
    print(f"Creating embeddings using HuggingFace model: {model}")
    
    embeddings = HuggingFaceEmbeddings(model_name=model)
    
    print(f"Embeddings model loaded successfully")
    
    embedded_chunks = embeddings.embed_documents([chunk.page_content for chunk in chunks])
    
    print(f"Total embeddings created: {len(embedded_chunks)}")
    if len(embedded_chunks) > 0:
        print(f"Sample embedding dimension: {len(embedded_chunks[0])}")
    
    return embeddings


def run_ingestion(doc_path="docs"):
    print("Ingestion Pipeline Started")
    
    # Loading Documents
    documents = load_docs(doc_path=doc_path)
    
    # Create Chunks
    chunks = create_chunks(documents)
    
    # Create Embeddings
    embeddings = create_embeddings(chunks)
    
    # Store in vector db
    db_path = "chroma_db"
    print(f"Storing chunks into Chroma vector database at {db_path}...")
    
    # Calculate deterministic IDs to prevent duplication on subsequent ingestions
    ids = [f"{chunk.metadata.get('source', 'unknown')}_chunk_{i}" for i, chunk in enumerate(chunks)]
        
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids,
        persist_directory=db_path
    )
    
    print("Ingestion complete. Documents are stored in vector database.")
    return vector_db

def main():
    run_ingestion()

if __name__ == "__main__":
    main()