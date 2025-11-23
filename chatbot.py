import os
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq
from langchain.embeddings.base import Embeddings


# -----------------------------
# CONFIG
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
CHROMA_PATH = "./chroma_db"


# -----------------------------
# Initialize models
# -----------------------------
# Whisper for STT
whisper_model = WhisperModel("small")

# ElevenLabs TTS Client
tts_client = ElevenLabs(api_key=ELEVEN_API_KEY)

# Groq LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")


# Custom Embedding class for Chroma
class MyEmbeddings(Embeddings):
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_query(self, text: str):
        return self.embedder.encode(text).tolist()

    def embed_documents(self, texts: list[str]):
        return self.embedder.encode(texts).tolist()


embedding_function = MyEmbeddings()


# Chroma Vector DB
vectorstore = Chroma(
    collection_name="insurance_faq",
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_function
)
retriever = vectorstore.as_retriever()


# RAG Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert insurance assistant. Use context to answer accurately."),
    ("human", "Chat history: {chat_history}\nContext: {context}\nQuestion: {question}")
])


# RAG Chain
rag_chain = (
    RunnableParallel(
        context=retriever,
        question=RunnablePassthrough()
    )
    | prompt
    | llm
)


# -----------------------------
# AUDIO FUNCTIONS
# -----------------------------
def record_audio(filename="input.wav", duration=5, samplerate=16000):
    print("🎙️ Recording... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    sf.write(filename, audio, samplerate)
    print("✅ Recording complete!")
    return filename


def transcribe_audio(filename):
    segments, _ = whisper_model.transcribe(filename)
    return " ".join(seg.text for seg in segments).strip()


def speak_text(text):
    try:
        audio_stream = tts_client.text_to_speech(
            voice="Rachel",
            model="eleven_multilingual_v2",
            text=text,
            stream=True
        )
        play(audio_stream)
    except Exception as e:
        print("❌ TTS Error:", e)




# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    chat_history = []

    print("🤖 Insurance Voice Chatbot Ready! (say 'exit')")

    while True:
        audio_file = record_audio()
        user_text = transcribe_audio(audio_file)
        print("📝 You said:", user_text)

        if user_text.lower() in ["exit", "quit", "bye"]:
            print("👋 Goodbye!")
            speak_text("Goodbye! Have a great day!")
            break

        history_str = "\n".join([f"User: {u}\nBot: {b}" for u, b in chat_history])

        try:
            result = rag_chain.invoke({
                "question": user_text,
                "chat_history": history_str
            })
            bot_reply = result.content
            chat_history.append((user_text, bot_reply))
        except Exception as e:
            print("❌ Bot Error:", e)
            bot_reply = "Sorry, something went wrong."

        print("🤖 Bot Reply:", bot_reply)
        speak_text(bot_reply)
