# indecimal-rag-chatbot
Here is your FINAL COMPLETE README.md file — just copy paste fully ✅

# 🏗️ Indecimal AI – RAG Chatbot

An AI-powered chatbot built using Retrieval-Augmented Generation (RAG) that answers user queries related to construction services, pricing, materials, and processes using a custom knowledge base.



## 🚀 Features

- 🔍 Semantic search using FAISS vector database  
- 🧠 Sentence Transformers for embeddings  
- 📄 Context-aware answering (no hallucinations)  
- 💬 Interactive chat UI using Streamlit  
- ⚡ Fast and efficient document retrieval  
- 🎛️ Adjustable Top-K chunk retrieval  
- 🔐 Secure API key handling using Streamlit Secrets  

## 🏗️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Embeddings:** Sentence Transformers (MiniLM / MPNet)  
- **Vector Database:** FAISS  
- **LLM:** OpenRouter (GPT-3.5 Turbo)  
- **Libraries:** LangChain, NumPy, Requests  

---

## 📂 Project Structure


indecimal-rag-chatbot/
│
├── app.py # Streamlit UI
├── doc1.md # Knowledge base
├── doc2.md
├── doc3.md
├── requirements.txt # Dependencies
├── README.md


## ⚙️ Installation (Run Locally)


git clone https://github.com/varshitha654/indecimal-rag-chatbot.git
cd indecimal-rag-chatbot

pip install -r requirements.txt
streamlit run app.py
🔐 API Key Setup (IMPORTANT)

Create a folder named .streamlit and inside it create a file:

.streamlit/secrets.toml

Paste your API key:

API_KEY = "your_openrouter_api_key"

Update your code:

API_KEY = st.secrets["API_KEY"]
🌐 Deployment (Streamlit Cloud)

Push your project to GitHub

Go to Streamlit Cloud

Click Deploy App

Select your repository

Add your API key in Settings → Secrets

Run your app 🚀

🧠 How It Works

Documents are loaded from .md files

Text is split into smaller chunks

Embeddings are generated using Sentence Transformers

Stored in FAISS vector database

User query is converted to embedding

Top-K relevant chunks are retrieved

LLM generates answer using retrieved context

💬 Example Questions

What is Indecimal?

How does pricing transparency work?

What services does Indecimal provide?

Do you provide real-time construction updates?

📌 Notes

Answers are generated strictly from the provided documents

If no relevant information is found, it returns "Not found"

Designed to reduce hallucination and improve accuracy

👩‍💻 Author

Varshitha
Electrical and Electronics Engineering
NIT Andhra Pradesh
