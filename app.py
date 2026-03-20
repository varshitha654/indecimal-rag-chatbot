# -------------------- 1. LOAD DOCUMENTS --------------------
from langchain_community.document_loaders import TextLoader

files = ["doc1.md", "doc2.md", "doc3.md"]
documents = []

for file in files:
    loader = TextLoader(file, encoding="utf-8")
    documents.extend(loader.load())

print("Total documents:", len(documents))


# -------------------- 2. CHUNKING --------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)
texts = [chunk.page_content.strip() for chunk in chunks]

print("Total chunks:", len(chunks))


# -------------------- 3. EMBEDDINGS --------------------
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')

embeddings = model.encode(texts, normalize_embeddings=True)

print("Embedding shape:", embeddings.shape)


# -------------------- 4. FAISS --------------------
import faiss

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)   # cosine similarity
index.add(np.array(embeddings))

print("FAISS index ready!")


# -------------------- 5. RETRIEVAL --------------------
def retrieve(query, k):
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, k)

    results = [texts[i] for i in indices[0]]

    # 🔥 hybrid keyword + semantic boost
    boosted = []
    for r in results:
        score = 0

        if "indecimal" in r.lower():
            score += 2

        if any(word in r.lower() for word in query.lower().split()):
            score += 1

        boosted.append((score, r))

    boosted = sorted(boosted, reverse=True)
    return [r for _, r in boosted]


# -------------------- 6. RERANK --------------------
def rerank(query, results):
    query_vec = model.encode([query], normalize_embeddings=True)[0]

    scored = []
    for text in results:
        text_vec = model.encode(text, normalize_embeddings=True)
        score = np.dot(query_vec, text_vec)
        scored.append((score, text))

    ranked = [x[1] for x in sorted(scored, reverse=True)]
    return ranked


# -------------------- 7. CLEAN CONTEXT --------------------
def clean_context(results):
    cleaned = []
    for r in results:
        r = r.replace("\n", " ").strip()
        if len(r) > 50:
            cleaned.append(r)
    return cleaned


# -------------------- 8. CONFIDENCE --------------------
def get_confidence(results):
    if len(results) >= 3:
        return "High"
    elif len(results) == 2:
        return "Medium"
    else:
        return "Low"


# -------------------- 9. LLM --------------------
import requests

API_KEY = "YOUR_API_KEY"   # 🔥 replace

def generate_answer(query, context):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are an expert assistant.

Instructions:
- Answer ONLY using the given context
- Provide a detailed explanation (5–8 sentences)
- First explain the concept clearly
- Then elaborate more details
- Then list key features as bullet points
- Combine multiple parts of context

- If partial information exists, still answer
- ONLY say "Not found" if context is completely unrelated

Context:
{context}

Question:
{query}

Answer:
"""

    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    if "choices" not in result:
        return f"API Error: {result}"

    return result["choices"][0]["message"]["content"]


# -------------------- 10. MAIN --------------------
query = "What are the key features of Indecimal?"

# 🔥 improved query expansion
expanded_query = f"""
{query}
indecimal construction platform services pricing transparency workflow benefits features
"""

# Retrieve
results = retrieve(expanded_query, k=6)

# Rerank
results = rerank(query, results)

# Clean
results = clean_context(results)

# Smart top-k
results = results[:min(4, len(results))]

# Confidence
confidence = get_confidence(results)

print("\nTop Retrieved Chunks:\n")
for i, res in enumerate(results):
    print(f"\nChunk {i+1}:\n{res}")

# Combine
context = "\n\n".join(results)

# Generate
if len(results) == 0 or len(context.strip()) < 30:
    answer = "Not found"
else:
    answer = generate_answer(query, context)

print("\nFinal Answer:\n")
print(answer)

print("\nConfidence:", confidence)