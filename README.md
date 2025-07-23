# 6-Conversational-ChatBot-RAG

Project Title: Conversational Q&A Chatbot with Retrieval-Augmented Generation (RAG)

Description:
Developed an interactive Conversational Q&A Chatbot leveraging Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers from both structured documents (PDFs, text files) and web resources. The system supports persistent user chat histories and session-based context, enabling seamless multi-turn conversations. Integrated with Streamlit for a user-friendly web interface, the chatbot utilizes advanced language models and vector search to retrieve and synthesize information in real time.

Key Technologies Used:
Python: Core programming language for backend logic and data processing.
Streamlit: For building the interactive web-based chat interface.
LangChain: Framework for orchestrating LLMs, retrieval, and conversational memory.
LLM (Groq Llama3-8b-8192): Large Language Model for generating responses.
HuggingFace Embeddings: For semantic vectorization of documents.
ChromaDB: Vector database for efficient document retrieval.
Document Loaders: Support for PDFs, text files, and web scraping (BeautifulSoup).
Session Management: Persistent chat history using JSON storage.
DuckDuckGo Search API: Optional tool for augmenting answers with web search results.
Environment Management: dotenv for secure API key and token handling.

Features:
Multi-turn conversational memory with user-specific chat history.
Retrieval from multiple sources: PDFs, text files, and web pages.
Real-time, context-aware Q&A with concise, relevant answers.
Modular, extensible architecture for easy integration of new data sources or models.
