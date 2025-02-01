
# ----------------------------
# Core System Dependencies
# ----------------------------
import os
import streamlit as st
from dotenv import load_dotenv

# ----------------------------
# AI Component Imports
# ----------------------------
# Vector Database Operations
from langchain_community.vectorstores import FAISS

# Conversation Chain Constructors
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Memory Management Components
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Prompt Engineering Tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LLM Providers Integration
from langchain_groq import ChatGroq

# Text Embedding Models
from langchain_huggingface import HuggingFaceEmbeddings

# Document Processing Utilities
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Conversation History Handlers
from langchain_core.runnables.history import RunnableWithMessageHistory

# Initialize environment configuration
load_dotenv()

# ----------------------------
# Embedding Model Setup
# ----------------------------
# Configure sentence embedding model for document vectorization
document_embedding_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',  # Optimized for balance between performance and speed
    model_kwargs={'device': 'cpu'}  # Force CPU execution for compatibility
)

# ----------------------------
# Streamlit Interface Configuration
# ----------------------------
st.set_page_config(page_title="Document Interaction Assistant", layout="wide")
st.header("📚 Document Interaction Assistant")
st.markdown("""
Transform static documents into conversational partners with our AI-powered interface.
**Key capabilities**:
- Context-aware Q&A from uploaded documents
- Continuous dialogue maintenance
- Intelligent content retrieval
""")

# ----------------------------
# System Documentation Sidebar
# ----------------------------
with st.sidebar:
    st.subheader("🖥️ System Overview")
    st.markdown("""
    **Next-gen Document Interaction Platform**  
    Leverage AI to transform static PDFs into interactive knowledge resources.  
    Combines state-of-the-art NLP with efficient document processing.
    """)
    
    st.subheader("🚀 How to Use")
    st.markdown("""
    1. **Authenticate** with Groq API key
    2. **Upload** PDF document
    3. **Ask questions** in natural language
    4. **Maintain dialogue** with follow-up queries
    5. **New session** = Fresh conversation context
    """)
    
    st.subheader("✨ Key Features")
    st.markdown("""
    - 📄 PDF document intelligence
    - 🔍 Semantic content retrieval
    - 💬 Context-aware conversations
    - 🧠 Short & long-term memory
    - ⚡ Real-time processing
    - 🔒 Local document processing
    """)
    
    st.subheader("⚙️ Operational Requirements")
    st.markdown("""
    - Groq Cloud API access
    - Hugging Face embeddings
    - Modern web browser
    - Stable internet connection
    """)

# ----------------------------
# User Authentication Module
# ----------------------------
llm_provider_key = st.text_input("Groq API Access Key", type='password')

if llm_provider_key:
    # ----------------------------
    # LLM Initialization
    # ----------------------------
    # Configure Groq's high-performance language model
    conversation_engine = ChatGroq(
        groq_api_key=llm_provider_key,
        model="deepseek-r1-distill-llama-70b",  # Optimized for document analysis
        temperature=0.3  # Balance creativity and factuality
    )

    # ----------------------------
    # Session Management System
    # ----------------------------
    session_id = st.text_input("Session Identifier", value="session_default")
    
    # Initialize session state container
    if 'system_memory' not in st.session_state:
        st.session_state.system_memory = {}

    # ----------------------------
    # Document Processing Pipeline
    # ----------------------------
    uploaded_file = st.file_uploader("PDF Document Upload", type="pdf")
    
    if uploaded_file:
        # ----------------------------
        # Temporary File Handling
        # ----------------------------
        temp_doc_path = "./temp_processing.pdf"
        with open(temp_doc_path, 'wb') as file_buffer:
            file_buffer.write(uploaded_file.getvalue())
            doc_title = uploaded_file.name

        # ----------------------------
        # Text Extraction & Chunking
        # ----------------------------
        pdf_loader = PyPDFLoader(temp_doc_path)
        raw_pages = pdf_loader.load()
        
        text_processor = RecursiveCharacterTextSplitter(
            chunk_size=4500,  # Optimized for LLM context windows
            chunk_overlap=600,  # Maintain contextual continuity
            length_function=len  # Standard character count
        )
        processed_chunks = text_processor.split_documents(raw_pages)

        # ----------------------------
        # Vector Knowledge Base Creation
        # ----------------------------
        vector_index = FAISS.from_documents(
            documents=processed_chunks,
            embedding=document_embedding_model
        )
        semantic_retriever = vector_index.as_retriever()

        # ----------------------------
        # Contextualization Subsystem
        # ----------------------------
        # Phase 1: Query Reformulation
        contextualization_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Analyze conversation history and current input. "
             "Generate standalone query if context-dependent, "
             "otherwise return original question."),
            MessagesPlaceholder("chat_chronology"),
            ("human", "{input}")
        ])

        # Create history-aware document finder
        contextual_retriever = create_history_aware_retriever(
            conversation_engine,
            semantic_retriever,
            contextualization_prompt
        )

        # ----------------------------
        # Response Generation Pipeline
        # ----------------------------
        # Phase 2: Knowledge-integrated Response
        response_template = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a technical document analyst. Use context to answer. "
             "If unsure, state 'No relevant information found'. "
             "Keep responses under 4 sentences.\nContext:\n{context}"),
            MessagesPlaceholder("chat_chronology"),
            ("human", "{input}")
        ])

        # Construct document-aware response chain
        response_assembler = create_stuff_documents_chain(
            conversation_engine,
            response_template
        )

        # ----------------------------
        # Integrated Retrieval System
        # ----------------------------
        knowledge_chain = create_retrieval_chain(
            contextual_retriever,
            response_assembler
        )

        # ----------------------------
        # Conversation Memory Manager
        # ----------------------------
        def memory_store(session_id: str) -> BaseChatMessageHistory:
            """Maintains conversation context across interactions"""
            if session_id not in st.session_state.system_memory:
                st.session_state.system_memory[session_id] = ChatMessageHistory()
            return st.session_state.system_memory[session_id]

        # ----------------------------
        # End-to-End Conversation System
        # ----------------------------
        conversational_agent = RunnableWithMessageHistory(
            knowledge_chain,
            memory_store,
            input_messages_key="input",
            history_messages_key="chat_chronology",
            output_messages_key="answer"
        )

        # ----------------------------
        # Chat Interface Management
        # ----------------------------
        if "message_log" not in st.session_state:
            st.session_state.message_log = []

        # Display conversation history
        for entry in st.session_state.message_log:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])

        # Process user input
        if user_query := st.chat_input("Document-related question"):
            # Update message history
            st.session_state.message_log.append({"role": "user", "content": user_query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_query)

            # Retrieve conversation context
            conversation_log = memory_store(session_id)

            # Generate AI response
            system_response = conversational_agent.invoke(
                {"input": user_query},
                config={"configurable": {"session_id": session_id}}
            )

            # Display and store response
            with st.chat_message("assistant"):
                st.markdown(system_response['answer'])
            
            st.session_state.message_log.append(
                {"role": "assistant", "content": system_response['answer']}
            )

else:
    st.warning("Authentication required: Provide Groq API key to initialize system")