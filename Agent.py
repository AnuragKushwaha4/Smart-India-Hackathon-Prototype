import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["HF_API_KEY"]=os.getenv("HF_API_KEY")

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from langchain_groq import ChatGroq
llm= ChatGroq(model="llama-3.3-70b-versatile")

#Tools creations:
# 1. Builtin tools:
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchResults
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain.agents import initialize_agent,AgentType

wikiAPI=WikipediaAPIWrapper(doc_content_chars_max=200,top_k_results=1)
arxivAPI = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)

wikipedia = WikipediaQueryRun(api_wrapper=wikiAPI)
arxiv = ArxivQueryRun(api_wrapper=arxivAPI)

search = DuckDuckGoSearchResults(name="Search_tool")

tools = [wikipedia,arxiv,search]


st.title("AI Search Agent")
st.sidebar.write("Upload WebLink:")
uploaded_webpage = st.sidebar.text_input("upload Link",placeholder="https://github.com")
if uploaded_webpage:
    #2.Custom Wrappers for searching from specific website.
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain.tools.retriever import create_retriever_tool
    loader = WebBaseLoader(web_path=uploaded_webpage)
    docs = loader.load()
    splitter =RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    splited_docs = splitter.split_documents(documents=docs)
    embeddings =HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector=Chroma.from_documents(documents=splited_docs,embedding=embeddings,)
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
    retriever,
    "Web_Based_tool",
    "Use this tool to answer questions using the content of the uploaded webpage. "
    "If the user provides a link, prefer this tool."
)

    tools.insert(0,retriever_tool)
    

if "message" not in st.session_state:
    st.session_state["message"]=[
        {
            "role":"Assistant","content":"Hi I am AI Agent who can search on web. How can i help you"
        }
    ]
#Printing Assistant's message for users:
for msg in st.session_state.message:
    st.chat_message(msg["role"]).write(msg['content'])

if uploaded_webpage:
    st.session_state.message.append({"role":"user","content":uploaded_webpage})
if prompt:=st.chat_input(placeholder="What is Generative AI?"):
    st.session_state.message.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    Agent = initialize_agent(tools,llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION)
# Printing response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        user_query = st.session_state.message[-1]["content"]
        response=Agent.run(st.session_state.message,callbacks=[st_cb])
        st.session_state.message.append({'role':'assistant', "content": response})
        st.write(response)

