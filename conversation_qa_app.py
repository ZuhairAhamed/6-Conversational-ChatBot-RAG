import streamlit as st
from conversation_qa import (
    get_llm,
    build_retriever,
    build_contextual_rag_chain,
    get_session_history_store,
    build_conversational_rag_chain,
)
import json
import os

HISTORY_FILE = "conversation_chat_histories.json"

def load_histories():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_histories(histories):
    with open(HISTORY_FILE, "w") as f:
        json.dump(histories, f)

# Initialize LLM, retriever, and chain only once (caching)
@st.cache_resource
def setup_chain():
    llm = get_llm()
    retriever = build_retriever()
    rag_chain = build_contextual_rag_chain(llm, retriever)
    store, get_session_history = get_session_history_store()
    conversational_rag_chain = build_conversational_rag_chain(rag_chain, get_session_history)
    return conversational_rag_chain, store

st.title("Conversational Q&A Chatbot (RAG)")

# User login/identification
username = st.text_input("Enter your username (or email):")
if not username:
    st.warning("Please enter your username to start chatting.")
    st.stop()
    
# Load all histories if not already loaded
if "all_histories" not in st.session_state:
    st.session_state.all_histories = load_histories()
    
# Use username as session_id
if "session_id" not in st.session_state or st.session_state.session_id != username:
    st.session_state.session_id = username
    st.session_state.chat_history = st.session_state.all_histories.get(username, [])
    

# Set up chain and store
conversational_rag_chain, store = setup_chain()

# Chat UI
st.markdown("Ask a question about lendo")

user_input = st.text_input("Your question:", key="input")

if st.button("Ask") and user_input.strip():
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Get answer from chain
    result = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )
    answer = result["answer"]

    # Add AI message to chat history
    st.session_state.chat_history.append({"role": "ai", "content": answer})

# When updating chat history:
st.session_state.all_histories[st.session_state.session_id] = st.session_state.chat_history
save_histories(st.session_state.all_histories)

# Display each user question as an expander, answer inside
user_msgs = [
    (i, msg["content"], st.session_state.chat_history[i+1]["content"])
    for i, msg in enumerate(st.session_state.chat_history)
    if msg["role"] == "user" and i+1 < len(st.session_state.chat_history) and st.session_state.chat_history[i+1]["role"] == "ai"
]

for i, (idx, question, answer) in enumerate(reversed(user_msgs)):
    with st.expander(f"Q{len(user_msgs)-i}: {question}", expanded=False):
        st.markdown(f"**AI:** {answer}")