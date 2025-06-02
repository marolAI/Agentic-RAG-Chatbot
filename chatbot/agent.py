import io
import os
import tempfile
from typing import List, Dict, Any, Literal
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from chatbot.utils import GradeDocuments, GRADE_PROMPT, REWRITE_PROMPT, GENERATE_PROMPT


class AgenticRAGSystem:
    AVAILABLE_MODELS = {
        "groq:meta-llama/llama-4-scout-17b-16e-instruct": "groq:meta-llama/llama-4-scout-17b-16e-instruct",
        "openai:gpt-4.1": "openai:gpt-4.1"
    }

    def __init__(
        self,
        api_key: str,
        model_name: str = "groq:meta-llama/llama-4-scout-17b-16e-instruct"
    ):
        """
        Initialize the agentic RAG system.
        
        Args:
            model_name: The name of model to use.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.llm = self._init_chat_model(self.api_key, self.AVAILABLE_MODELS[self.model_name])
        self.retriever = None  # Will be initialized later
        self.graph = None  # Will be built after document is loaded

    def _init_chat_model(self, api_key: str, model_name: str):
        """Initialize the chat model
        """
        return init_chat_model(model_name, api_key=api_key, temperature=0)
    
    def load_document(self, file: io.BytesIO) -> None:
        """
        Load and process a PDF document
        
        Args:
            file: Uploaded file object
        """
        self.retriever = self.get_retriever(file)
        self.retriever_tool = self._create_retriever_tool()
        self.graph = self.build_graph()  # Build graph after document is loaded

    def get_retriever(
        self, 
        file: io.BytesIO,
    ) -> VectorStoreRetriever:
        """
        Get the retriever.
        
        Args:
            file_path: The path to the document file.
            
        Returns:
            The retriever.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getbuffer())
            file_path = tmp_file.name
        try:
            loader = PyPDFLoader(file_path, extract_images=True)
            splitted_docs = loader.load_and_split()
            print(splitted_docs)
            embeddings = HuggingFaceEmbeddings(model_kwargs={'device': 'cpu'})
            vector_store = DocArrayInMemorySearch.from_documents(splitted_docs, embeddings)
        finally:
            os.unlink(file_path)
        return vector_store.as_retriever()
    
    def _create_retriever_tool(self) -> Tool:
        """
        Create retriever tool.
        
            
        Returns:
            The retriever tool.
        """
        return create_retriever_tool(
                self.retriever,
                "retrieve_docs",
                "Search and return relevant information about the documents."
            )
    
    def generate_query_or_respond(self, state: MessagesState) -> Dict[str, Any]:
        """
        Decide whether to retrieve context or respond directly.
        
        Args:
            state: Current agent state.
            
        Returns:
            Dictionary with action and parameters.
        """
        response = (self.llm.bind_tools([self.retriever_tool]).invoke(state["messages"]))
        return {"messages": [response]}
    
    def grade_documents(
            self, 
            state: MessagesState
        ) -> Literal["generate_answer", "rewrite_question"]:
        """
        Evaluate document relevance to the question.
        
        Args:
            state: Current agent state.
            
        Returns:
            Next node name ('generate_answer' or 'rewrite_question')
        """
        question = state["messages"][0].content
        context = state["messages"][-1].content

        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = (
            self.llm
            .with_structured_output(GradeDocuments).invoke(
                [{"role": "user", "content": prompt}]
            )
        )
        score = response.binary_score

        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"

    def rewrite_question(self, state: MessagesState) -> Dict[str, List[Dict]]:
        """
        Improve the original question based on retrieval failure.
        
        Args:
            state: Current agent state.
            
        Returns:
            Rewritten question string.
        """
        messages = state["messages"]
        question = messages[0].content
        prompt = REWRITE_PROMPT.format(question=question)
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [{"role": "user", "content": response.content}]}
    
    def generate_answer(self, state: MessagesState) -> Dict[str, List[Dict]]:
        """
        Generate final response using question and context.
        
        Args:
            question: User question (original or rewritten).
            context: Relevant documents.
            
        Returns:
            Final answer string.
        """
        question = state["messages"][0].content
        context = state["messages"][-1].content
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = self.llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}
    
    def build_graph(self) -> StateGraph:
        """Construct the agent workflow graph"""
        builder = StateGraph(state_schema=MessagesState)

        # Define the nodes we will cycle between
        builder.add_node(self.generate_query_or_respond)
        builder.add_node("retrieve", ToolNode([self.retriever_tool]))
        builder.add_node(self.rewrite_question)
        builder.add_node(self.generate_answer)

        builder.add_edge(START, "generate_query_or_respond")

        # Decide whether to retrieve
        builder.add_conditional_edges(
            "generate_query_or_respond",
            # Assess LLM decision (call `retriever_tool` tool or respond to the user)
            tools_condition,
            {
                # Translate the condition outputs to nodes in our graph
                "tools": "retrieve",
                END: END,
            },
        )

        builder.add_conditional_edges(
            "retrieve",
            self.grade_documents,
        )
        builder.add_edge("generate_answer", END)
        builder.add_edge("rewrite_question", "generate_query_or_respond")

        return builder.compile(checkpointer=MemorySaver())
    
    
    def chat(self, message: str, thread_id: str):
        return self.graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            {"configurable": {"thread_id": thread_id}}
        )
    
    def update_model(self, model_name: str):
        self.model_name = model_name
        self.llm = self._init_chat_model(self.api_key, self.AVAILABLE_MODELS[self.model_name])
        self.app = self.build_graph()