# # """
# # Conversation Q&A Chatbot (Refactored)
# # -------------------------------------
# # A modular script for conversational question-answering using retrieval-augmented generation (RAG) with chat history support.
# # """
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import glob
import bs4
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Load environment variables
load_dotenv()

def get_llm():
    """Initialize and return the LLM (Groq) instance."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

def get_embeddings():
    """Initialize and return HuggingFace embeddings."""
    os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def read_urls_from_txt():
    """Read URLs from a text file, one per line, and return as a list."""
    file_path = "urls.txt"
    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    return urls


class_name  = ["features-hero-subtext-darkbg", "features-pointer-heading", "features-pointer-subheading" , "point-line-item",
 "how-it-works-item-subtext-invest", "features-point-text", "home-split-subtext-3", "home-split-subtext-4", "notice-wrapper"]


def build_duckduckgo_tool(max_results=1):
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
    return DuckDuckGoSearchRun(name="duckduck-tool", api_wrapper=wrapper)

def build_retriever():
    
    docs = []
    
    # Load PDF files
    pdf_files = glob.glob("resources/*.pdf")
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs.extend(loader.load())
        
    # Load TXT files
    txt_files = glob.glob("resources/*.txt")
    for txt_file in txt_files:
        loader = TextLoader(txt_file)
        docs.extend(loader.load())
    
    # Load website content
    urls = read_urls_from_txt()
    loader = WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=class_name
            )
        ),
    )
    docs.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()

    return retriever
    # return create_retriever_tool(
    #     retriever, 
    #     "retriver-tool", 
    #     "Searches the loaded PDF, TXT, and website resources for relevant information to answer user questions."
    #     )

def get_system_prompt():
    return (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )

def get_contextualize_q_prompt():
    return (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

def build_rag_chain(llm, retriever):
    """Build a basic RAG chain (no chat history)."""
    system_prompt = get_system_prompt()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

def build_contextual_rag_chain(llm, retriever):
    """Build a RAG chain with chat history support."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", get_contextualize_q_prompt()),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt()),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_session_history_store():
    """Return a session-based chat history store and getter."""
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    return store, get_session_history

def build_conversational_rag_chain(rag_chain, get_session_history):
    """Wrap the RAG chain with message history for conversational use."""
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

def example_usage():
    """Demonstrate usage of the conversational RAG chain."""
    urls = read_urls_from_txt()
    
    # retriever_tool = build_retriever_tool()
    # duckduckgo_tool = build_duckduckgo_tool()
    # tools = [retriever_tool, duckduckgo_tool]
    
    # llm = get_llm().bind_tools(tools)
    
    llm = get_llm()
    retriever = build_retriever()

    rag_chain = build_contextual_rag_chain(llm, retriever)
    store, get_session_history = get_session_history_store()
    conversational_rag_chain = build_conversational_rag_chain(rag_chain, get_session_history)
    session_id = "abc123"
    print("Q: What is Task Decomposition?")
    answer1 = conversational_rag_chain.invoke(
        {"input": "What is Task Decomposition?"},
        config={"configurable": {"session_id": session_id}}
    )["answer"]
    print("A:", answer1)
    print("Q: What are common ways of doing it?")
    answer2 = conversational_rag_chain.invoke(
        {"input": "What are common ways of doing it?"},
        config={"configurable": {"session_id": session_id}}
    )["answer"]
    print("A:", answer2)
    # Optionally return the store for inspection
    return store

if __name__ == "__main__":
    example_usage() 