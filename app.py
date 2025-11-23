import os 
from io import BytesIO
from dotenv import load_dotenv
import gradio as gr
from elevenlabs import ElevenLabs

# Embeddings + Vector Store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# LLM
from langchain_groq import ChatGroq

# Memory
from langchain_core.messages import HumanMessage, AIMessage



# -------------------- LOAD API KEYS --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

tts_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)


# -------------------- VECTOR STORE --------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# -------------------- LLM + MEMORY --------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant"
)



# -------------------- HYBRID QA FUNCTION --------------------
def answer_question(user_text):
    """
    If PDF contains relevant answer → use RAG
    Else → fallback to Groq LLM general knowledge
    """
    try:
        # Search vector database
        docs = vectorstore.similarity_search(user_text, k=2)

        # If similarity search returned nothing → fallback to LLM
        if not docs:
            return llm.invoke(user_text).content

        # If context is not relevant enough → fallback to LLM
        # (Basic relevance check)
        # NOTE: Chroma doesn't provide distance in metadata,
        # so we check text overlap instead — simple but effective.
        context_text = " ".join([d.page_content for d in docs])
        if len(set(user_text.lower().split()) & set(context_text.lower().split())) < 2:
            return llm.invoke(user_text).content

        # If relevant → RAG answer
        prompt = f"Use the following context to answer:\n\n{context_text}\n\nQuestion: {user_text}"
        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        return f"Error in processing: {e}"


# -------------------- AUDIO PIPELINE --------------------
def bot_pipeline(audio_file, history):
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()

        # Speech → Text
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            user_text = recognizer.recognize_google(audio)

        # Get answer (Hybrid RAG)
        bot_reply = answer_question(user_text)

        # Text → Voice
        response = tts_client.text_to_speech.convert(
            voice_id="FGY2WhTYpPnrIDTdsKH5",
            model_id="eleven_multilingual_v2",
            text=bot_reply,
        )

        audio_stream = BytesIO()
        for chunk in response:
            if chunk:
                audio_stream.write(chunk)
        audio_stream.seek(0)

        output_path = "bot_reply.mp3"
        with open(output_path, "wb") as f:
            f.write(audio_stream.read())

        # Update UI Chat History
        history.append((user_text, bot_reply))
        return history, output_path

    except Exception as e:
        history.append(("Error", str(e)))
        return history, None


# -------------------- GRADIO UI --------------------
with gr.Blocks() as demo:
    gr.Markdown("## 💬 Insurance Voice Chatbot")

    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak here")
            send_btn = gr.Button("Send")
        
        with gr.Column():
            chatbot = gr.Chatbot(label="Conversation", height=400, bubble_full_width=False)
            audio_out = gr.Audio(label="🔊 Bot Reply", type="filepath")

    send_btn.click(
        fn=bot_pipeline,
        inputs=[audio_in, chatbot],
        outputs=[chatbot, audio_out]
    )

demo.launch()
