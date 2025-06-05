import os
import time
import torch
import streamlit as st

from chatbot.agent import AgenticRAGSystem
from chatbot.utils import create_sidebar, StreamHandler

torch.classes.__path__ = [] # RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!


st.set_page_config(page_icon="ℹ️", page_title="RAGi")
st.header("**Hi, I am 🅡🅐🅖ℹ️.**")
st.write("What can I help you today?")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = None
if "chatbot" not in st.session_state:
    st.session_state.chatbot = AgenticRAGSystem(st.secrets["GROQ_API"])

create_sidebar()

prompt = st.chat_input(accept_file=True, file_type=["pdf"], placeholder="Ask me anything!")

if prompt is not None:
    # Extract components
    user_query = prompt.text
    uploaded_files = prompt.get("files", [])
    
    # Validation flags
    has_valid_query = user_query and user_query.strip() != ""
    has_valid_file = len(uploaded_files) > 0
    document_loaded = st.session_state.get("document_loaded")
    
    # Handle valid input
    if has_valid_query and (has_valid_file or document_loaded):
        # Process new file if uploaded
        if has_valid_file:
            uploaded_file = uploaded_files[0]
            print(document_loaded, uploaded_file.name)
            if document_loaded != uploaded_file.name:
                with st.spinner("Processing document..."):
                    st.session_state.chatbot.load_document(uploaded_file)
                    st.session_state.document_loaded = uploaded_file.name
                    st.success(f"Loaded: {uploaded_file.name}")
        
        with st.chat_message("assistant"):
            st_cb = StreamHandler(st.empty())
            result = st.session_state.chatbot.graph.invoke(
                    {"messages": [{"role": "user", "content": user_query}]},
                    {
                        "configurable": {"thread_id": str(time.time())},
                        "recursion_limit": 100,
                        "callbacks": [st_cb]
                    }
                )
            response = result["messages"][-1].content
            st.session_state.messages.append({"role": "assistant", "content": response})

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])
    
    # Handle invalid cases
    else:
        if not has_valid_query:
            st.error("Please enter a question to continue!")
        if not has_valid_file and not document_loaded:
            st.error("Please upload a PDF document to continue!")
