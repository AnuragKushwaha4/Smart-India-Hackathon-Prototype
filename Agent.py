import os
import tempfile
import streamlit as st

# --- Environment Setup ---
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["HF_API_KEY"] = st.secrets["HF_API_KEY"]

# --- Imports ---
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain.agents.react import create_react_agent
from langchain.agents import RunnableAgent
from langchain import hub


# --- Initialize LLM ---
llm = ChatGroq(model="llama-3.3-70b-versatile")

# --- Builtin Tools ---
wikiAPI = WikipediaAPIWrapper(doc_content_chars_max=200, top_k_results=1)
arxivAPI = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)

wikipedia = WikipediaQueryRun(api_wrapper=wikiAPI)
arxiv = ArxivQueryRun(api_wrapper=arxivAPI)
search = DuckDuckGoSearchResults(name="Search_tool")

tools = [wikipedia, arxiv, search]

# --- Streamlit UI ---
st.title("AI Agent")
st.sidebar.title("Hello..")
st.sidebar.write(
    "I’m your AI Assistant, an intelligent AI agent designed to help you navigate university resources, "
    "answer academic questions, and provide insights from uploaded study materials or official documents. "
    "You can ask me about courses, assignments, research papers, or any information from AUM’s website, "
    "and I’ll give you concise, context-aware answers. How can I assist you today?"
)

upload_file = st.file_uploader("Upload PDF File:", type="pdf", accept_multiple_files=False)

# --- PDF File Processing ---
if upload_file:
    from langchain_community.document_loaders import PyPDFium2Loader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain.tools.retriever import create_retriever_tool

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(upload_file.getbuffer())
        tmp_path = tmp_file.name

    loader = PyPDFium2Loader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splited_docs = splitter.split_documents(documents=docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector = Chroma.from_documents(documents=splited_docs, embedding=embeddings)
    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "Document_tool",
        "Use this tool to answer questions using the content of the uploaded PDF. "
        "If the user provides a link, prefer this tool."
    )

    tools.insert(0, retriever_tool)

# --- Session Handling ---
if "message" not in st.session_state:
    st.session_state["message"] = [
        {"role": "assistant", "content": "Hi I am AI Agent who can search on web. How can I help you?"}
    ]

for msg in st.session_state.message:
    st.chat_message(msg["role"]).write(msg["content"])

if upload_file:
    st.session_state.message.append({"role": "user", "content": f"Uploaded file: {upload_file.name}"})


# --- Agent Creation ---
prompt_template = hub.pull("hwchase17/react")
react_agent = create_react_agent(llm, tools, prompt=prompt_template)
Agent = RunnableAgent(agent=react_agent, tools=tools)

# --- Chat Logic ---
if prompt_text := st.chat_input(placeholder="What is Generative AI?"):
    st.session_state.message.append({"role": "user", "content": prompt_text})
    st.chat_message("user").write(prompt_text)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = Agent.invoke({"input": prompt_text}, callbacks=[st_cb])
            output = response.get("output", "No response generated.")
        except Exception as e:
            output = f"⚠️ Error: {e}"
        st.session_state.message.append({'role': 'assistant', "content": output})
        st.write(output)
