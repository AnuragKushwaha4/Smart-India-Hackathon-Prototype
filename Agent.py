import os
import streamlit as st

os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["HF_API_KEY"] = st.secrets["HF_API_KEY"]


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq

st.title("RAG Document With Memory History")
session_id = st.text_input("Session State",value="Default")

if "store" not in st.session_state:
    st.session_state.store={}

upload_file= st.file_uploader("Upload File:",type="pdf",accept_multiple_files=True)

if upload_file:
    llm= ChatGroq(model="llama-3.3-70b-versatile")
    document =[]
    for uploaded_files in upload_file:
        path =f"./Resources/temp.pdf"
        with open(path,"wb") as f:
            f.write(uploaded_files.getvalue())
        loader= PyPDFLoader(path)
        docs = loader.load()
        document.extend(docs)

    splitter =RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    splited_docs = splitter.split_documents(documents=document)
    embeddings =HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector=Chroma.from_documents(documents=splited_docs,embedding=embeddings,)

    retriever = vector.as_retriever()

    contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    history_aware_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user","{input}")
        ]
    )

    retriever_with_message_history = create_history_aware_retriever(llm=llm,retriever=retriever,prompt=history_aware_prompt)


    systemPrompt=(
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",systemPrompt),
            MessagesPlaceholder("chat_history"),
            ("user","{input}")
        ]
    )
    document_chain = create_stuff_documents_chain(prompt=prompt,llm=llm)
    ragChain = create_retrieval_chain(retriever_with_message_history,document_chain)

    def getsessionHistory(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    
    Converstionalchain =RunnableWithMessageHistory(
        ragChain,getsessionHistory,
        input_messages_key="input",
        output_messages_key="answer",
        history_messages_key="chat_history"
    )
    input_text=st.text_input("Enter your Query..")
    if input_text:
        res=Converstionalchain.invoke(
            {"input":input_text},
            config={
                    "configurable": {"session_id":session_id}
                }
        )
        st.write(res['answer'])