# 📚 RAG Learning Notes — Day 01
**Date:** 17 October 2025  
**Topic:** RAG kya hota hai? Embeddings & Cosine Similarity  
**Language:** Hinglish (Easy mode ON 😎)

---

## 🤔 RAG kya hota hai? (Simple explanation)

**RAG = Retrieval Augmented Generation**

Bhai, seedha simple bhasha mein samajh:

> LLM (jaise ChatGPT) ke paas **apna knowledge** hota hai, lekin woh **teri personal info** nahi jaanta.  
> RAG us problem ko solve karta hai — **pehle relevant info dhoondo, phir LLM ko do.**

```
User Question
     |
     ↓
[1. RETRIEVE]  →  Apne documents/chunks mein se best match dhoondo
     |
     ↓
[2. AUGMENT]   →  Question + Retrieved context = ek bada prompt banao
     |
     ↓
[3. GENERATE]  →  LLM ko wo prompt do → Answer milta hai
```

### Real life example 🍕
Socho tere paas ek pizza restaurant ka menu hai (documents).  
Customer poochta hai: *"Kya vegetarian options hain?"*

- **Without RAG:** LLM apne training se guess karega (galat bhi ho sakta hai)
- **With RAG:** Menu se vegetarian items dhoond ke LLM ko deta hai → Accurate answer ✅

---

## 🔢 Embeddings kya hote hain?

**Embedding = Text ko numbers (vectors) mein convert karna**

LLM words nahi samajhta directly — woh **numbers** samajhta hai.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "raj delhi mein rehta hai"
embedding = model.encode(text)

# Output: [0.23, -0.45, 0.12, 0.89, ...]  ← 384 numbers ka array
print(embedding.shape)  # (384,)
```

### Visualization 📊

```
"I love coding"   →  [0.2,  0.8, -0.1, 0.5, ...]
"I enjoy coding"  →  [0.19, 0.79, -0.09, 0.52, ...]   ← similar!
"I hate pizza"    →  [-0.5, 0.1,  0.9,  -0.3, ...]   ← different
```

**Key point:** Similar meaning = similar numbers = vectors ek doosre ke **paas** honge!

---

## 📐 Cosine Similarity kya hoti hai?

Yeh measure karta hai ki **do vectors kitne similar hain**.

```
Score = 1.0  →  Ekdum same (twin bhai 👯)
Score = 0.5  →  Thoda similar
Score = 0.0  →  Bilkul alag (strangers 🙅)
Score = -1.0 →  Opposite meaning
```

### Angle diagram:

```
          B (query vector)
         /
        /  ← angle θ (chhota angle = zyada similar)
       /
      /_________ A (chunk vector)
     
θ chhota = cos(θ) bada = zyada similar ✅
θ bada   = cos(θ) chhota = kam similar ❌
```

### Code mein:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

query_emb = model.encode(["where does raj live?"])
chunk_embs = model.encode([
    "my name raj.",
    "raj is from delhi.",   # ← yeh sabse similar hoga!
    "i love data science."
])

scores = cosine_similarity(query_emb, chunk_embs)
# scores = [[0.31, 0.89, 0.12]]

best = np.argmax(scores[0])
print(chunks[best])  # "raj is from delhi."
```

---

## 🏗️ Hamara RAG Pipeline — Step by Step

```
Documents/Chunks (apna knowledge base)
  "my name raj."
  "raj is from delhi."
  "i am 20 year old."
          |
          ▼ model.encode()
  Embeddings (numbers)
  [[0.2, -0.1, 0.8, ...],
   [0.5,  0.3, 0.1, ...],   ← best match
   [0.1, -0.5, 0.9, ...]]
          |
  User Question ──► encode ──► Query Embedding
          |
          ▼ cosine_similarity()
  Best Match: "raj is from delhi."
          |
          ▼ Groq LLM (prompt mein dalo)
  Final Answer: "Raj Delhi mein rehta hai."
```

---

## 📁 Project Structure

```
rag-learning/
├── .env              ← 🔒 SECRET (gitignore, push nahi hoga)
├── .env.example      ← ✅ Template (yeh push hoga)
├── .gitignore
├── rag_mini_demo.py  ← Main RAG code
├── demo_emedings.py  ← Embeddings demo
└── notes/
    └── day01_rag_basics.md  ← Yeh file!
```

---

## 🔑 Environment Variables — Kya aur Kyun?

**Problem:** API key directly code mein likhna = DANGEROUS 🚨

```python
# ❌ GALAT — kabhi mat karo
client = Groq(api_key="gsk_abc123...")  # GitHub par visible!
```

**Solution:** `.env` file mein daalo, `os.getenv()` se lo

```python
# ✅ SAHI
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
```

**.env file (local only, never push):**
```
GROQ_API_KEY=gsk_abc123...
```

**.gitignore mein:**
```
.env
.venv/
```

---

## 🛠️ Tools Used Aaj

| Tool | Kaam | Free? |
|------|------|-------|
| `sentence-transformers` | Text → Embeddings | ✅ Yes |
| `all-MiniLM-L6-v2` | Lightweight embedding model | ✅ Yes |
| `sklearn cosine_similarity` | Similarity calculate karna | ✅ Yes |
| `Groq API` | Fast LLM inference | ✅ Free tier |
| `openai/gpt-oss-20b` | LLM model on Groq | ✅ Available |
| `python-dotenv` | .env se keys load karna | ✅ Yes |

---

## 💡 Aaj ke Key Takeaways

1. **RAG = Search + Generate** — pehle relevant info dhoondo, phir answer lo
2. **Embeddings** — text ko numbers mein convert karte hain comparison ke liye
3. **Cosine Similarity** — 0 to 1 ke beech similarity score
4. **API keys** — kabhi hardcode mat karo, `.env` use karo
5. **`all-MiniLM-L6-v2`** — fast aur free embedding model

---

## 🚀 Kal Kya Karein? (Next Steps)

- [ ] Multiple documents se RAG banana (file pado → chunks banao)
- [ ] FAISS ya ChromaDB use karke proper vector database banana
- [ ] PDF se text extract karke RAG chalana
- [ ] Top-K chunks retrieve karna (sirf 1 nahi, top 3)

---

*Notes by: Raj | RAG Learning Journey 🚀*
