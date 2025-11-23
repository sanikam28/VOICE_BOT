import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---- Load PDF ----
PDF_FILE = "policy pdf.pdf"
loader = PyPDFLoader(PDF_FILE)
docs = loader.load()

# ---- Split ----
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# ---- Embeddings + Chroma ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# ---- Groq LLM ----
llm = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"
)

# ---- New RAG Chain ----
prompt = PromptTemplate(
    template="""You are an insurance assistant.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

retriever = db.as_retriever(search_kwargs={"k": 3})

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# ---- Chat Loop ----
print("🤖 Insurance Chatbot is ready! (type 'exit' to quit)")

while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("👋 Goodbye!")
        break

    answer = rag_chain.invoke(query)
    print("\nAssistant:", answer.content)
