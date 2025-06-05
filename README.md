# RAGi: Agentic RAG Chatbot

## 🅡🅐🅖ℹ️

RAGi is an intelligent document understanding system powered by advanced Retrieval Augmented Generation (RAG) techniques and an agentic architecture. It's designed to answer complex questions based on your documents by autonomously deciding the best approach, verifying information, and providing responses grounded in specific sources.

## Features

*   **Autonomous Context Retrieval:** Intelligently decides when and how to retrieve relevant information from documents rather than relying solely on the language model's parametric memory.
*   **Self-Correction & Query Refinement:** Features a built-in mechanism to identify potential issues with retrieval or answers and refine the query or approach for better results.
*   **Source Verification:** Verifies the relevance and truthfulness of retrieved document chunks before incorporating them into the final answer.
*   **Grounded Responses with Citations:** Prevents hallucinations by ensuring all generated answers are directly supported by the source documents, often allowing for citation or reference back to the source text.
*   **Intelligent Document Parsing:** Handles various document formats and can extract text from image-based content (like scanned PDFs) using OCR.

## Tech Stack

This project leverages the following key technologies and libraries to achieve its agentic and RAG capabilities:

*   **Python:** The foundational programming language.
*   **Streamlit:** Provides the interactive and user-friendly web interface.
*   **Langchain & LangGraph:** Core frameworks for building robust LLM applications. Langchain orchestrates the RAG chains and interactions with various models, while LangGraph is essential for defining the stateful, autonomous agent workflows that enable decision-making and self-correction.
*   **Groq API**: Provides the underlying large language model (LLM) inference capabilities for summarization and text analysis.

## Quick Start

1.  Clone the repository:
    ```bash
    git clone https://github.com/marolAI/Agentic-RAG-Chatbot.git
    cd Agentic-RAG-Chatbot
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up necessary environment variables (e.g., API keys for Groq, potentially others).
4.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
5.  Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

## License

This project is licensed under the [MIT License](license.txt).

## Contact

For questions or feedback, please contact the author:

*   **Author:** Andriamarolahy R.
*   **GitHub:** `https://github.com/marolAI`
*   **Email: marolahyrabe@gmail.com**