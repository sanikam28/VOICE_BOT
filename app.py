import os
from io import BytesIO
from dotenv import load_dotenv
import gradio as gr
from elevenlabs import ElevenLabs
import speech_recognition as sr

# LangChain Components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq


# -------------------- ENVIRONMENT SETUP --------------------
def load_api_clients():
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    eleven_key = os.getenv("ELEVENLABS_API_KEY")
    
    tts_client = ElevenLabs(api_key=eleven_key)
    llm_client = ChatGroq(groq_api_key=groq_key, model="llama-3.1-8b-instant")
    
    return tts_client, llm_client


# -------------------- SPEECH RECOGNITION --------------------
def transcribe_audio(audio_file_path):
    """Convert speech to text using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
    except Exception as e:
        return f"Error in speech recognition: {e}"


# -------------------- VECTOR STORE SETUP --------------------
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)


# -------------------- HYBRID QA (RAG + LLM) --------------------
def answer_question(user_text, llm, vectorstore):
    """Hybrid approach: use RAG if relevant context exists, else fallback to LLM."""
    try:
        docs = vectorstore.similarity_search(user_text, k=2)

        # No docs found → use LLM
        if not docs:
            return llm.invoke(user_text).content

        context_text = " ".join([d.page_content for d in docs])

        # Basic relevance check (word overlap)
        if len(set(user_text.lower().split()) & set(context_text.lower().split())) < 2:
            return llm.invoke(user_text).content

        prompt = f"Use the following context to answer:\n\n{context_text}\n\nQuestion: {user_text}"
        return llm.invoke(prompt).content

    except Exception as e:
        return f"RAG/LLM Error: {e}"


# -------------------- TEXT TO SPEECH --------------------
def synthesize_speech(text, tts_client):
    """Convert bot text reply into an audio file."""
    try:
        response = tts_client.text_to_speech.convert(
            voice_id="FGY2WhTYpPnrIDTdsKH5",
            model_id="eleven_multilingual_v2",
            text=text,
        )

        audio_stream = BytesIO()
        for chunk in response:
            if chunk:
                audio_stream.write(chunk)
        audio_stream.seek(0)

        output_path = "bot_reply.mp3"
        with open(output_path, "wb") as f:
            f.write(audio_stream.read())

        return output_path

    except Exception as e:
        return f"Error in text-to-speech: {e}"


# -------------------- MAIN BOT PIPELINE --------------------
def bot_pipeline(audio_file, history):
    user_text = transcribe_audio(audio_file)

    if "Error" in user_text:
        history.append(("Error", user_text))
        return history, None

    bot_reply = answer_question(user_text, llm, vectorstore)
    audio_output = synthesize_speech(bot_reply, tts_client)

    history.append((user_text, bot_reply))
    return history, audio_output


# -------------------- INITIALIZE GLOBAL CLIENTS --------------------
tts_client, llm = load_api_clients()
vectorstore = load_vectorstore()


# -------------------- GRADIO UI --------------------
def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown("## 💬 Insurance Voice Chatbot")

        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak here")
                send_btn = gr.Button("Send")

            with gr.Column():
                chatbot = gr.Chatbot(label="Conversation", height=400, bubble_full_width=False)
                audio_out = gr.Audio(label="🔊 Bot Reply", type="filepath")

        send_btn.click(fn=bot_pipeline, inputs=[audio_in, chatbot], outputs=[chatbot, audio_out])

    demo.launch()


if __name__ == "__main__":
    launch_app()
