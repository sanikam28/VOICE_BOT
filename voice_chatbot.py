import os
from dotenv import load_dotenv
import speech_recognition as sr
from elevenlabs import ElevenLabs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pydub import AudioSegment
from pydub.playback import play as pydub_play
from io import BytesIO

# ================== CONFIG ==================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

tts_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
recognizer = sr.Recognizer()

# ================== RECORD AUDIO ==================
def record_audio(timeout=5, phrase_time_limit=10):
    print("🎙 Speak now...")
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("⌛ No speech detected")
            return None

    try:
        text = recognizer.recognize_google(audio)
        print("📝 You said:", text)
        return text
    except:
        print("❌ Could not understand")
        return None

# ================== TTS ==================
def speak_text(text: str):
    try:
        response = tts_client.text_to_speech.convert(
            text=text,
            voice_id="FGY2WhTYpPnrIDTdsKH5",          # voice name or ID
            model_id="eleven_multilingual_v2"
        )

        audio_bytes = b"".join(response)   # join streaming chunks

        audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
        pydub_play(audio_segment)

    except Exception as e:
        print("❌ TTS Error:", e)


# ================== RAG SETUP ==================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./vectorstore",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are an insurance assistant. Use ONLY the context below.

<context>
{context}
</context>

Question: {question}
""")

# Runnable chain (LATEST LC VERSION)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ================== MAIN LOOP ==================
print("🤖 Voice bot ready! Say 'exit' to stop.")

while True:
    user_text = record_audio()

    if not user_text:
        continue

    if user_text.lower() in ["exit", "quit", "bye", "stop"]:
        print("👋 Goodbye!")
        break

    try:
        answer = rag_chain.invoke(user_text)
        print("🤖 Bot:", answer)
        speak_text(answer)
    except Exception as e:
        print("❌ Bot Error:", e)
