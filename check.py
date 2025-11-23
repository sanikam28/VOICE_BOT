from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Define embedding model (recommended for accuracy)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load saved Chroma DB
vectorstore = Chroma(
    persist_directory="chroma",
    embedding_function=embeddings
)

# Search in DB
docs = vectorstore.similarity_search(
    "documents required to file a claim?",
    k=2
)

for doc in docs:
    print("\n📄 Retrieved Document:\n", doc.page_content)
