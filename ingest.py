import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_pdf(filepath):
    """Load PDF document."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    loader = PyPDFLoader(filepath)
    return loader.load()


def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def create_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Initialize embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_db(chunks, embeddings, persist_directory="./chroma_db"):
    """Create and persist Chroma vector database."""
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return db


def ingest_pdf_to_vector_db(pdf_path, persist_directory="./chroma_db"):
    """Main pipeline to process PDF and build vector DB."""
    print("📥 Loading PDF...")
    docs = load_pdf(pdf_path)

    print("✂️ Splitting into chunks...")
    chunks = split_documents(docs)

    print("🧠 Generating embeddings...")
    embeddings = create_embeddings()

    print("💾 Creating Vector Database...")
    db = create_vector_db(chunks, embeddings, persist_directory)

    print("🎉 Ingestion complete! Vector DB created at:", persist_directory)
    return db


if __name__ == "__main__":
    PDF_FILE = "policy pdf.pdf"
    ingest_pdf_to_vector_db(PDF_FILE)
