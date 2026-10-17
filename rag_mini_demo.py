
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()  # .env file se keys load karta hai

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = [
    "my name raj.",
    "i am a btech student.",
    "raj is from delhi.",
    "i am 20 year old.",
    "currently learn rag.",
    "my favourite subject is data science."
]

# Convert chunks into embeddings
embeddings = model.encode(chunks)

# User question
question = "where does raj live?"

# Convert question into embedding
query_embedding = model.encode([question])

# Find similarity
scores = cosine_similarity(query_embedding, embeddings)

# Get the most similar chunk
best_index = np.argmax(scores[0])

context = chunks[best_index]

print("Retrieved context:", context)

# Connect to Groq
client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # .env file se API key load hogi

# Create prompt
prompt = f"""
Answer the question using context information explain answer like friend in hinglish no extra word use kar context ke hisab se or any other extra words please.

Context: {context}

Question: {question}
"""

# Send context and question to LLM
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("LLM Answer:", response.choices[0].message.content)
