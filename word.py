import os
import json
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# ----------------------------------------------------
# Step 1: Load GROQ API Key
# ----------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")
print("🔑 GROQ API key loaded from .env")

# ----------------------------------------------------
# Step 2: Load dataset
# ----------------------------------------------------
print("📦 Loading dataset...")
with open("evaluation_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"✅ Dataset loaded. {len(data)} samples found.")

# ----------------------------------------------------
# Step 3: Load vectorstore (optional for retrieval)
# ----------------------------------------------------
print("🔍 Loading vectorstore...")
CHROMA_PATH = "./chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
print("✅ Vectorstore loaded.")

# ----------------------------------------------------
# Step 4: Initialize ChatGroq LLM with wrapper
# ----------------------------------------------------
print("🔄 Initializing ChatGroq LLM...")
base_llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Wrap the LLM for RAGAS compatibility
wrapped_llm = LangchainLLMWrapper(base_llm)
wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)

print("✅ ChatGroq LLM ready.")

# ----------------------------------------------------
# Step 5: Prepare data in RAGAS format
# ----------------------------------------------------
print("✏️ Preparing data for RAGAS evaluation...")

ragas_data = {
    'question': [],
    'answer': [],
    'context': [],
    'ground_truth': []
}

for item in data:
    question = item.get("question", "")
    ground_truth = item.get("ground_truth", "")
    
    # Get context - convert to list if it's a string
    context = item.get("context", "")
    if isinstance(context, str):
        contexts = [context]
    else:
        contexts = context if context else [""]
    
    # Generate answer using your chatbot
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    try:
        response = base_llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        print(f"❌ Error generating response for question: {question}")
        print(f"   Error: {e}")
        answer = ""
    
    ragas_data['question'].append(question)
    ragas_data['answer'].append(answer)
    ragas_data['context'].append(context)
    ragas_data['ground_truth'].append(ground_truth)

# Create RAGAS dataset
dataset = Dataset.from_dict(ragas_data)
print(f"✅ Data prepared. {len(dataset)} samples ready for evaluation.")

# ----------------------------------------------------
# Step 6: Run RAGAS evaluation with wrapped LLM
# ----------------------------------------------------
print("\n📊 Running RAGAS evaluation...")
print("This may take a few minutes...\n")

try:
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings
    )
    
    # ----------------------------------------------------
    # Step 7: Display and save results
    # ----------------------------------------------------
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Display individual metric scores
    for metric_name, score in results.items():
        if isinstance(score, (int, float)):
            print(f"{metric_name.replace('_', ' ').title()}: {score:.4f}")
    
    print("="*60)
    
    # Convert to DataFrame
    results_df = results.to_pandas()
    
    # Save detailed results
    output_file = "evaluation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Detailed results saved to {output_file}")
    
    # Show sample of detailed results
    print("\nSample of detailed results:")
    display_cols = [col for col in ['question', 'answer_relevancy', 'faithfulness', 
                    'context_precision', 'context_recall'] if col in results_df.columns]
    print(results_df[display_cols].head())
    
    # Analyze problem areas
    print("\n" + "="*60)
    print("AREAS FOR IMPROVEMENT")
    print("="*60)
    
    threshold = 0.7
    issues_found = False
    
    if 'faithfulness' in results_df.columns:
        low_faithfulness = results_df[results_df['faithfulness'] < threshold]
        if not low_faithfulness.empty:
            issues_found = True
            print(f"\n⚠️  {len(low_faithfulness)} questions with low faithfulness (<{threshold})")
            print("   → Your chatbot may be hallucinating or adding info not in context")
    
    if 'answer_relevancy' in results_df.columns:
        low_relevancy = results_df[results_df['answer_relevancy'] < threshold]
        if not low_relevancy.empty:
            issues_found = True
            print(f"\n⚠️  {len(low_relevancy)} questions with low answer relevancy (<{threshold})")
            print("   → Answers may not directly address the questions")
    
    if 'context_precision' in results_df.columns:
        low_precision = results_df[results_df['context_precision'] < threshold]
        if not low_precision.empty:
            issues_found = True
            print(f"\n⚠️  {len(low_precision)} questions with low context precision (<{threshold})")
            print("   → Irrelevant contexts ranked too high")
    
    if 'context_recall' in results_df.columns:
        low_recall = results_df[results_df['context_recall'] < threshold]
        if not low_recall.empty:
            issues_found = True
            print(f"\n⚠️  {len(low_recall)} questions with low context recall (<{threshold})")
            print("   → Missing important contexts during retrieval")
    
    if not issues_found:
        print("\n✅ All metrics above threshold! Great job!")
    
    print("\n" + "="*60)

except Exception as e:
    print(f"\n❌ Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    print("\nTroubleshooting tips:")
    print("1. Make sure ragas is up to date: pip install --upgrade ragas")
    print("2. Check your evaluation_data.json format")
    print("3. Try with OpenAI instead: pip install openai")