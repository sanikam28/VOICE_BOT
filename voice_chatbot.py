import os
import sys
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
from pydub.playback import play
from io import BytesIO


# ================== CONFIG ==================
def load_config():
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    eleven_key = os.getenv("ELEVENLABS_API_KEY")

    if not groq_key or not eleven_key:
        print("❌ Missing API keys in .env file.")
        sys.exit(1)

    return groq_key, eleven_key


# ================== SPEECH RECOGNITION ==================
class SpeechRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def record_audio(self, timeout=5, phrase_time_limit=10):
        print("🎙 Speak now...")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
            except sr.WaitTimeoutError:
                print("⌛ No speech detected")
                return None

        return self.transcribe_audio(audio)

    def transcribe_audio(self, audio):
        try:
            text = self.recognizer.recognize_google(audio)
            print("📝 You said:", text)
            return text
        except sr.UnknownValueError:
            print("❌ Could not understand speech")
            return None
        except Exception as e:
            print(f"❌ Speech Recognition Error: {e}")
            return None


# ================== TEXT TO SPEECH ==================
class TextToSpeech:
    def __init__(self, api_key):
        self.client = ElevenLabs(api_key=api_key)

    def speak(self, text: str):
        try:
            response = self.client.text_to_speech.convert(
                text=text,
                voice_id="FGY2WhTYpPnrIDTdsKH5",
                model_id="eleven_multilingual_v2"
            )
            audio_bytes = b"".join(response)
            audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
            play(audio_segment)
        except Exception as e:
            print(f"❌ TTS Error: {e}")


# ================== RAG BOT ==================
class RAGVoiceBot:
    def __init__(self):
        # Embeddings & Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            persist_directory="./vectorstore",
            embedding_function=embeddings
        )
        self.retriever = vectorstore.as_retriever()

        # LLM
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1
        )

        # Prompt
        self.prompt = ChatPromptTemplate.from_template("""
        You are an insurance assistant. Use ONLY the context below.

        <context>
        {context}
        </context>

        Question: {question}
        """)

        # RAG Chain
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, user_input: str) -> str:
        try:
            response = self.chain.invoke(user_input)
            return response
        except Exception as e:
            print(f"❌ RAG Error: {e}")
            return "I'm sorry, I couldn't process your question."


# ================== MAIN ==================
def main():
    _, eleven_api_key = load_config()

    speech_engine = SpeechRecognizer()
    tts_engine = TextToSpeech(eleven_api_key)
    bot = RAGVoiceBot()

    print("🤖 Voice bot ready! Say 'exit' to stop.")

    while True:
        user_text = speech_engine.record_audio()

        if not user_text:
            continue

        if user_text.lower() in ["exit", "quit", "bye", "stop"]:
            print("👋 Goodbye!")
            break

        answer = bot.query(user_text)
        print("🤖 Bot:", answer)
        tts_engine.speak(answer)


if __name__ == "__main__":
    main()
