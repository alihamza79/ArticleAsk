import os
import streamlit as st
import pickle
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Define file path for storing processed data
file_path = "vector_news"

load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="News Research Assistant",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2E4053;
    }
    .stAlert {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📰 News Research Assistant")
st.markdown("### Analyze multiple news articles with AI-powered insights")

# Improved sidebar
with st.sidebar:
    st.markdown("### 📑 Add News Articles")
    st.info("Please enter the URLs of the news articles you want to analyze (up to 3)")
    
    urls = []
    for i in range(3):
        url = st.text_input(f"Article URL #{i+1}", key=f"url_{i}")
        if url:
            urls.append(url)

    process_url_clicked = st.button("🔍 Process Articles", type="primary")
    
    if os.path.exists(file_path):
        st.success("✅ Previous analysis data available")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool helps you analyze multiple news articles using AI. "
               "Enter the URLs and ask questions about the content.")

# Main content area
main_placeholder = st.empty()

# Initialize LLM with required params
llm = ChatGroq(
    temperature=0.9,
    max_tokens=500,
    model_name="llama-3.3-70b-versatile"  
)

if process_url_clicked:
    if not urls:
        st.error("Please enter at least one URL to process")
    else:
        with st.spinner("Processing articles..."):
            progress_bar = st.progress(0)
            
            # Loading data
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.info("📥 Loading articles...")
            data = loader.load()
            progress_bar.progress(0.3)
            
            # Splitting text
            main_placeholder.info("📑 Processing text...")
            text_spliter = RecursiveCharacterTextSplitter(
                separators=['\n\n','\n','.',','],
                chunk_size=1000
            )
            docs = text_spliter.split_documents(data)
            progress_bar.progress(0.6)
            
            # Embedding
            main_placeholder.info("🔄 Creating embeddings...")
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorIndex_hugging = FAISS.from_documents(docs, embedding_model)
            progress_bar.progress(0.9)
            
            # Save using FAISS native method
            vectorIndex_hugging.save_local(file_path)
            
            progress_bar.progress(1.0)
            st.success("✅ Articles processed successfully!")
            main_placeholder.empty()

# Query section
if os.path.exists(file_path + ".faiss"):
    st.markdown("---")
    st.markdown("### Ask Questions About the Articles")
    query = st.text_input("Enter your question:", placeholder="e.g., What are the main topics discussed?")
    
    if query:
        with st.spinner("Analyzing..."):
            # Load FAISS index
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorStore = FAISS.load_local(file_path, embedding_model)
            
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            # Display answer in a card-like container
            st.markdown("### 💡 Answer")
            with st.container():
                st.markdown(f">{result['answer']}")
            
            # Display sources in an expander
            with st.expander("View Sources"):
                for source in result["sources"].split("\n"):
                    if source.strip():
                        st.markdown(f"- {source.strip()}")

# Footer
st.markdown("---")
st.markdown("Built by Ali Hamza Don")
