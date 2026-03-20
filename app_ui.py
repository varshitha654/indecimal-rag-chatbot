import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Indecimal AI", page_icon="logo.png", layout="wide")

st.markdown("""
<style>

.main { background-color: #0e1117; }

/* Sidebar theme */
section[data-testid="stSidebar"] {
    background-color: #161b22;
}

/* Slider color */
.stSlider [role="slider"] {
    background-color: #a34a3c !important;
}
.stSlider div[data-baseweb="slider"] div {
    color: #a34a3c !important;
}

/* Subtitle */
.subtitle {
    color: #9ca3af;
}

/* Chat container */
.chat-container {
    max-width: 800px;
    margin: auto;
    padding-bottom: 80px;
}

/* Chat layout */
.chat-row {
    display: flex;
    align-items: flex-end;
    margin: 10px 0;
}

.user-row { justify-content: flex-end; }
.bot-row { justify-content: flex-start; }

/* Avatar */
.avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    margin: 0 8px;
    border: 2px solid #a34a3c;
}

/* Chat bubble */
.chat-bubble {
    padding: 10px 14px;
    border-radius: 12px;
    max-width: 70%;
    font-size: 15px;
}

/* Theme */
.user-msg { background-color: #a34a3c; color: white; }
.bot-msg { background-color: #1b1f26; color: white; }

/* Buttons */
div.stButton > button {
    background-color: #111827;
    color: white;
    border-radius: 20px;
    padding: 8px 14px;
    border: 1px solid #5b2c24;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    background-color: #c97a6a;
    border: 1px solid #c97a6a;
    box-shadow: 0 0 8px rgba(201,122,106,0.5);
}

/* FAQ row */
.faq-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    margin-top: 20px;
}

.faq-title {
    color: white;
    font-weight: 600;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
col1, col2 = st.columns([0.6, 10])

with col1:
    st.markdown("<div style='margin-top:10px;'>", unsafe_allow_html=True)
    st.image("logo.png", width=55)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <h2 style="color:white; margin:0;">
        Indecimal AI Assistant
    </h2>
    <p style="color:#9ca3af; margin:2px 0 0 0;">
        Ask anything about construction, pricing, materials & process
    </p>
    """, unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("### Indecimal AI")
st.sidebar.markdown("---")

st.sidebar.markdown("### ⚙️ Settings")

k = st.sidebar.slider("Top K chunks", 1, 10, 5)

show_context = st.sidebar.toggle("Show Retrieved Context", True)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Info")
st.sidebar.write("Embedding: MiniLM")
st.sidebar.write("LLM: GPT-3.5")

# -------------------- LOAD DATA --------------------
@st.cache_resource
def load_data():
    files = ["doc1.md", "doc2.md", "doc3.md"]
    documents = []

    for file in files:
        loader = TextLoader(file, encoding="utf-8")
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return texts, model, index

texts, model, index = load_data()

# -------------------- RETRIEVE --------------------
def retrieve(query, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# -------------------- LLM --------------------
API_KEY = st.secrets["API_KEY"]

def generate_answer(query, context):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Answer ONLY from the context below.
If answer not present, say "Not found".

Context:
{context}

Question:
{query}
"""

    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    if "choices" not in result:
        return f"API Error: {result}"

    return result["choices"][0]["message"]["content"]

# -------------------- CHAT MEMORY --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.session_state.messages = st.session_state.messages[-10:]

# -------------------- HANDLE QUESTION --------------------
def handle_question(question):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking... 🤔"):
        results = retrieve(question, k)
        context = "\n\n".join(results)
        answer = generate_answer(question, context)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "context": results
    })

# -------------------- FAQ --------------------
st.markdown('<div class="faq-container">', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([2,2,2,2])

with col1:
    st.markdown('<div class="faq-title">🔍 Most Asked</div>', unsafe_allow_html=True)

with col2:
    if st.button("What is Indecimal?"):
        handle_question("What is Indecimal and what services do you provide?")

with col3:
    if st.button("Pricing transparency?"):
        handle_question("How does Indecimal ensure pricing transparency?")

with col4:
    if st.button("Real-time updates?"):
        handle_question("Do you provide real-time construction updates?")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------- INPUT --------------------
query = st.chat_input("Type your question here...")

if query:
    handle_question(query)

# -------------------- DISPLAY CHAT --------------------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-row user-row">
            <div class="chat-bubble user-msg">{msg["content"]}</div>
            <img src="https://cdn-icons-png.flaticon.com/512/847/847969.png" class="avatar">
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="chat-row bot-row">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png" class="avatar">
            <div class="chat-bubble bot-msg">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

        if show_context:
            with st.expander("📄 Retrieved Context"):
                for i, res in enumerate(msg.get("context", [])):
                    st.markdown(f"**Chunk {i+1}:**")

                    st.markdown(f"""
                    <div style="
                        background-color:#3b1f1a;
                        color:white;
                        padding:12px;
                        border-radius:10px;
                        margin-bottom:10px;
                        border:1px solid #a34a3c;
                    ">
                    {res}
                    </div>
                    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
