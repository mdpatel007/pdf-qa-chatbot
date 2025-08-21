# 🧠 PDF Q&A Chatbot

A smart, Streamlit-powered chatbot that reads and answers questions from uploaded PDF documents using **Groq's LLMs** and **LangChain**. Built for recruiters, researchers, and anyone who wants instant insights from documents.

## 🚀 Live Demo

👉 [Click here to try it live](https://your-streamlit-app-url.com)  
*(Replace with your actual Streamlit Cloud URL)*

---

## 📦 Features

- 📄 Upload any PDF and ask questions about its content
- ⚡ Powered by Groq's blazing-fast LLMs via LangChain
- 🧠 Intelligent chunking and embedding for accurate answers
- 🛡️ Secrets managed via `.env` (Groq API key not exposed)
- 🎯 Clean UI with Streamlit for recruiter-friendly presentation

---

## 🛠️ Tech Stack

| Tool        | Purpose                          |
|-------------|----------------------------------|
| Python      | Core programming language        |
| Streamlit   | Frontend UI                      |
| LangChain   | PDF parsing + LLM orchestration  |
| Groq API    | LLM backend                      |
| FAISS       | Vector store for embeddings      |
| PyPDF2      | PDF text extraction              |

---

## 📁 Project Structure

pdf_qa_bot/ 
├── app.py # Main Streamlit app 
├── requirements.txt # Dependencies 
├── README.md # Project overview


---

## 🔐 Environment Setup

Create a `.env` file in the root directory:

GROQ_API_KEY=your_groq_api_key_here

git clone https://github.com/mdpatel007/pdf-qa-chatbot.git
cd pdf-qa-chatbot
pip install -r requirements.txt
streamlit run app.py

## 📚 How It Works
1. PDF Upload: User uploads a PDF file.
2. Text Extraction: PDF is split into chunks using LangChain.
3. Embedding: Chunks are embedded and stored in FAISS.
4. Query: User asks a question.
5. LLM Response: Groq LLM retrieves relevant chunks and answers.

## 🧠 Example Use Cases
- Recruiters scanning resumes
- Students summarizing research papers
- Lawyers reviewing contracts
- Analysts extracting insights from reports

## 🙌 Credits
Built by Mihir Dudhat Inspired by real-world document analysis needs and powered by cutting-edge LLMs.
