import os
import streamlit as st

# -------------------- API Keys --------------------
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["HF_API_KEY"] = st.secrets["HF_API_KEY"]

# -------------------- Imports --------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI  # Replace with ChatGroq if installed
from langchain.chains import RetrievalQA
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

# -------------------- Streamlit UI --------------------
st.title("RAG Document With Memory History")
session_id = st.text_input("Session State", value="Default")

if "store" not in st.session_state:
    st.session_state.store = {}

upload_file = st.file_uploader("Upload File:", type="pdf", accept_multiple_files=True)

if upload_file:
    # Load PDF documents
    documents = []
    for uploaded_file in upload_file:
        path = f"./Resources/temp.pdf"
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splited_docs = splitter.split_documents(documents)

    # Create embeddings + vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(splited_docs, embeddings)

    # Retriever
    retriever = vector_store.as_retriever()

    # Chat model
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # -------------------- RAG Chain --------------------
    system_prompt = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, say that you don't know.
    Keep answers concise (max 3 sentences). 
    {context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    rag_chain = RetrievalQA.from_chain(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    # -------------------- Session Memory --------------------
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = []

    input_text = st.text_input("Enter your Query..")
    if input_text:
        # Add previous chat history
        chat_history = st.session_state.store[session_id]
        res = rag_chain.run({"input": input_text, "chat_history": chat_history})
        chat_history.append({"user": input_text, "assistant": res})
        st.session_state.store[session_id] = chat_history
        st.write(res)
