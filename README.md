# VOICE_BOT
Dependencies

This project uses the following major libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| langchain | 0.3.11 | Core RAG pipeline |
| langchain-community | 0.3.11 | Tools, loaders, vectorstore |
| langchain-huggingface | 0.1.2 | Embedding model integration |
| sentence-transformers | 2.7.0 | Embedding model (MiniLM-L6-v2) |
| chromadb | 0.5.23 | Vector database for retrieval |
| elevenlabs | 1.50.3 | Text-to-speech synthesis |
| groq | 0.9.1 | Groq LLM client |
| SpeechRecognition | 3.12.0 | Speech-to-text |
| PyAudio | 0.2.14 | Microphone input support |
| pydub | 0.25.1 | Audio processing |
| python-dotenv | 1.0.1 | Environment variable loader |
| requests | 2.32.3 | API requests |

PDF STRUCTURE

voice-rag-assistant
│── app.py
│── voice_chatbot.py
│── ingest.py
│── requirements.txt
│── .env.example
│── README.md
│
├── vectorstore/
├── pdf_docs/

WEBSITES TO GENERATE API KEYS

GROQ API Key:
https://console.groq.com

ElevenLabs API Key:
https://elevenlabs.io
