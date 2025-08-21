import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --------------------------
# API Key (Groq)
# --------------------------
api_key = "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"

if not api_key:
    st.error("⚠️ GROQ_API_KEY not found! Please set it as an environment variable.")
else:
    st.success("🔑 API Key Loaded")

# --------------------------
# Streamlit UI
# --------------------------
st.title("📄 PDF Q&A Chatbot (Groq + LangChain)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# --------------------------
# Keyword Highlighter
# --------------------------
def highlight_keywords(text, keywords):
    for kw in keywords:
        text = text.replace(kw, f"<span style='color:#1f77b4; font-weight:bold'>{kw}</span>")
    return text

keywords = [
    "Python", "OCR", "EDA", "GridSearchCV", "Streamlit", "Scikit-learn",
    "Random Forest", "Regex", "Docker", "MongoDB", "Firebase", "NLP", "Computer Vision"
]

if uploaded_file:
    # --------------------------
    # Step 1: Read PDF
    # --------------------------
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # --------------------------
    # Step 2: Split Text
    # --------------------------
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(text)

    # --------------------------
    # Step 3: Embeddings & Vector DB
    # --------------------------
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    retriever = knowledge_base.as_retriever()

    # --------------------------
    # Step 4: Load Groq LLM
    # --------------------------
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

    # --------------------------
    # Step 5: Prompt Template
    # --------------------------
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided documents to answer questions."),
        ("human", "Context:\n{context}\n\nQuestion: {input}")
    ])

    # --------------------------
    # Step 6: Retrieval Chain
    # --------------------------
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # --------------------------
    # Step 7: Chat History
    # --------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --------------------------
    # Step 8: User Question
    # --------------------------
    user_question = st.text_input("💬 Ask a question about your PDF:")

    if user_question:
        try:
            response = retrieval_chain.invoke({"input": user_question})
            st.session_state.chat_history.append((user_question, response["answer"]))
        except Exception as e:
            st.error(f"⚠️ Model error: {str(e)}")

    # --------------------------
    # Step 9: Display Chat History with Highlighting
    # --------------------------
    for q, a in st.session_state.chat_history:
        st.write(f"**🧠 Q:** {q}")
        highlighted = highlight_keywords(a, keywords)
        st.markdown(f"**📌 A:** {highlighted}", unsafe_allow_html=True)

    # --------------------------
    # Step 10: Resume Summary + Download
    # --------------------------
    if st.button("🧾 Summarize Resume"):
        try:
            summary = retrieval_chain.invoke({"input": "Summarize this resume in 5 bullet points."})
            st.write("### 📝 Summary:")
            highlighted_summary = highlight_keywords(summary["answer"], keywords)
            st.markdown(highlighted_summary, unsafe_allow_html=True)

            st.download_button(
                label="📥 Download Summary",
                data=summary["answer"],
                file_name="mihir_resume_summary.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"⚠️ Summary error: {str(e)}")

    # --------------------------
    # Step 11: Recruiter Mode
    # --------------------------
    recruiter_mode = st.checkbox("🎯 Recruiter Mode")

    if recruiter_mode:
        try:
            recruiter_summary = retrieval_chain.invoke({
                "input": "Summarize Mihir Dudhat's resume in 3 bullet points for a recruiter."
            })
            st.write("### 💼 Recruiter Summary:")
            highlighted_recruiter = highlight_keywords(recruiter_summary["answer"], keywords)
            st.markdown(highlighted_recruiter, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Recruiter summary error: {str(e)}")

    # --------------------------
    # Step 12: Interview Prep Mode 
    # --------------------------
    interview_prep = st.checkbox("🗣️ Interview Prep Mode")

    if interview_prep:
        st.info("🧪 Interview Prep Mode is under development. In future, this will generate behavioral questions based on your resume content.")
        st.write("**Example Questions:**")
        st.markdown("- Tell me about a time you built a scalable system from scratch.")
        st.markdown("- How did you handle model tuning in your churn prediction project?")
        st.markdown("- Describe a situation where you used OCR to solve a real-world problem.")
