from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2")
chunks=[
    "hello my name is raj chaudhary.",
    " i am a btech student.",
    "iam from delhi.",
    "iam 20 year old.",
    "currently learn rag."
]

emeddings=model.encode(chunks)
print()