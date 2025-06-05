import os
from datetime import datetime
import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.callbacks import BaseCallbackHandler


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

class StreamHandler(BaseCallbackHandler):
    
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)


def add_space(num_spaces=1):
    for _ in range(num_spaces):
        st.write("\n")


def create_sidebar():
    with st.sidebar:
        st.header("🅡🅐🅖ℹ️", divider="gray")
        st.markdown("""
        **Intelligent document understanding** powered by advanced retrieval augmented generation techniques.
        This system understands your documents and answers complex questions with citations.
        """)
        
        st.markdown(
            """
                [![source code](https://img.shields.io/badge/view_source_code-gray?logo=github)](https://github.com/marolAI/Agentic-RAG-Chatbot/blob/main/app.py)
                
                🔍[More projects](https://marolai.github.io/projects/)
            """
        )
        st.divider()
        add_space(20)
        
        
        st.divider()
        
        st.caption("Built with ❤️ using Streamlit, Langgraph, and Groq")
        st.caption(f"v1.0 | {datetime.now().year}")
