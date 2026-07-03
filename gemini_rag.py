"""
DOCUQUERY GEMINI - Zero-install RAG System
Uses Google's Gemini API (100% free, no model downloads)
"""

import os
import hashlib
from pathlib import Path
import time
from datetime import datetime

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from pypdf import PdfReader
from dotenv import load_dotenv

# ============================================
# CONFIGURATION
# ============================================

load_dotenv()

class Config:
    """All settings in one place"""
    
    # Gemini settings
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = "gemini-2.5-flash"
    
    # Paths
    CHROMA_PATH = "./chroma_db"
    COLLECTION_NAME = "documents"
    
    # Chunking
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval
    TOP_K_RESULTS = 5
    
    # System prompt
    SYSTEM_PROMPT = """You are a precise document Q&A assistant. 

RULES:
1. ONLY use information from the provided context
2. If the context doesn't contain the answer, say: "I cannot find this information in the uploaded document."
3. Cite the source filename and page number when possible
4. Be concise but thorough
5. Never invent or hallucinate information

CONTEXT:
{context}

QUESTION: {question}

ANSWER (based only on the context above):"""

# Initialize Gemini
if Config.GEMINI_API_KEY:
    genai.configure(api_key=Config.GEMINI_API_KEY)

# ============================================
# DOCUMENT PROCESSING
# ============================================

class DocumentProcessor:
    """Handles PDF loading and chunking"""
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """Extract text from PDF with page numbers"""
        try:
            reader = PdfReader(file_path)
            text = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text.append(f"[Page {i+1}]\n{page_text}")
            return "\n\n".join(text)
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    @staticmethod
    def smart_chunk(text: str, chunk_size: int = Config.CHUNK_SIZE, 
                   overlap: int = Config.CHUNK_OVERLAP) -> list:
        """Split text into overlapping chunks"""
        chunks = []
        
        # Split by pages first
        pages = text.split("\n\n[Page ")
        
        for page_content in pages:
            if not page_content.strip():
                continue
            
            # Extract page number
            page_num = 1
            if page_content.startswith("]"):
                try:
                    page_num = int(page_content[1:].split("]")[0])
                    page_content = page_content.split("]", 1)[1].strip()
                except:
                    pass
            
            # Split into chunks
            words = page_content.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                current_length += len(word) + 1
                if current_length > chunk_size and current_chunk:
                    # Save chunk
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "page": page_num,
                        "source": None  # Will be set later
                    })
                    
                    # Keep overlap words
                    overlap_words = current_chunk[-overlap:] if overlap > 0 else []
                    current_chunk = overlap_words
                    current_length = sum(len(w) + 1 for w in overlap_words)
                
                current_chunk.append(word)
            
            # Last chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "page": page_num,
                    "source": None
                })
        
        return chunks
    
    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Generate unique ID for file"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:12]

# ============================================
# VECTOR DATABASE
# ============================================

class VectorDatabase:
    """ChromaDB with sentence-transformers embeddings"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_fn = None
        self.initialized = False
    
    def initialize(self):
        """Setup ChromaDB"""
        try:
            self.client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
            
            # Use sentence-transformers for embeddings
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            self.collection = self.client.get_or_create_collection(
                name=Config.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            
            self.initialized = True
            return True
        except Exception as e:
            st.error(f"Database init failed: {str(e)}")
            return False
    
    def add_documents(self, chunks: list, file_name: str, file_hash: str):
        """Add chunks to vector DB"""
        if not self.initialized:
            self.initialize()
        
        # Prepare data
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": file_name,
                "file_hash": file_hash,
                "page": chunk["page"],
                "chunk_id": i
            }
            for i, chunk in enumerate(chunks)
        ]
        ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
        
        # Add to ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(chunks)
    
    def search(self, query: str, k: int = Config.TOP_K_RESULTS) -> list:
        """Search for relevant chunks"""
        if not self.initialized:
            self.initialize()
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": 1 - results['distances'][0][i] if results['distances'] else 0
                })
        
        return formatted
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        if not self.initialized:
            self.initialize()
        
        count = self.collection.count()
        
        sources = set()
        if count > 0:
            results = self.collection.get(limit=100)
            for meta in results['metadatas']:
                if meta and 'source' in meta:
                    sources.add(meta['source'])
        
        return {
            "total_chunks": count,
            "documents": len(sources),
            "doc_names": list(sources)[:5]
        }

# ============================================
# GEMINI LLM INTERFACE
# ============================================

class GeminiInterface:
    """Handles all Gemini API interactions"""
    
    @staticmethod
    def check_api():
        """Verify API key works"""
        if not Config.GEMINI_API_KEY:
            return False, "No API key found"
        
        try:
            model = genai.GenerativeModel(Config.GEMINI_MODEL)
            response = model.generate_content("Hello")
            return True, "API working"
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def generate_answer(question: str, context_chunks: list) -> dict:
        """Generate answer using Gemini"""
        
        # Prepare context
        context_parts = []
        for chunk in context_chunks:
            source = chunk['metadata']['source']
            page = chunk['metadata']['page']
            text = chunk['text'][:800]  # Limit for token efficiency
            context_parts.append(f"[Source: {source}, Page {page}]\n{text}")
        
        full_context = "\n\n---\n\n".join(context_parts)
        
        # Format prompt
        prompt = Config.SYSTEM_PROMPT.format(
            context=full_context,
            question=question
        )
        
        try:
            # Call Gemini
            start_time = time.time()
            model = genai.GenerativeModel(Config.GEMINI_MODEL)
            response = model.generate_content(prompt)
            elapsed = time.time() - start_time
            
            return {
                "answer": response.text,
                "time": elapsed,
                "model": Config.GEMINI_MODEL,
                "context_used": len(context_chunks)
            }
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "time": 0,
                "model": Config.GEMINI_MODEL,
                "context_used": 0
            }

# ============================================
# STREAMLIT UI
# ============================================

def main():
    """Main application"""
    
    st.set_page_config(
        page_title="DocuQuery",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    if 'db' not in st.session_state:
        st.session_state.db = VectorDatabase()
        st.session_state.db.initialize()
    
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.title("📚 DocuQuery")
        
        
        # API Key status
        st.subheader("🔑 API Status")
        
        if not Config.GEMINI_API_KEY:
            st.error("❌ Gemini API key not found")
            
            # Show input for API key
            api_key = st.text_input("Enter your Gemini API key:", 
                                   type="password",
                                   help="Get free key at aistudio.google.com")
            
            if api_key:
                # Save to .env
                with open(".env", "w") as f:
                    f.write(f"GEMINI_API_KEY={api_key}")
                st.success("✅ API key saved! Please restart the app.")
                st.rerun()
        else:
            api_status, api_message = GeminiInterface.check_api()
            if api_status:
                st.success(f"✅ Gemini API active")
            else:
                st.error(f"❌ API error: {api_message}")
        
        st.divider()
        
        # Database stats
        stats = st.session_state.db.get_stats()
        st.info(f"📊 Database: {stats['total_chunks']} chunks from {stats['documents']} docs")
        
        st.divider()
        
        # ===== Upload Section =====
        st.subheader("📤 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            key="pdf_uploader"
        )
        
        if uploaded_file:
            # Save file
            Path("./uploads").mkdir(exist_ok=True)
            save_path = f"./uploads/{uploaded_file.name}"
            
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("🔄 Process Document", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    # Extract text
                    text = DocumentProcessor.load_pdf(save_path)
                    
                    if text:
                        # Chunk
                        chunks = DocumentProcessor.smart_chunk(text)
                        
                        # Add source names
                        for chunk in chunks:
                            chunk['source'] = uploaded_file.name
                        
                        # Hash and store
                        file_hash = DocumentProcessor.get_file_hash(save_path)
                        count = st.session_state.db.add_documents(
                            chunks, 
                            uploaded_file.name,
                            file_hash
                        )
                        
                        st.success(f"✅ Added {count} chunks!")
                        st.rerun()
        
        st.divider()
        
        # Reset button
        if st.button("🗑️ Clear All Data", type="secondary"):
            try:
                st.session_state.db.client.delete_collection(Config.COLLECTION_NAME)
                st.session_state.db.initialize()
                st.session_state.conversation = []
                st.success("Database cleared!")
                st.rerun()
            except:
                pass
    
    # ========== MAIN CONTENT ==========
    
    st.title("🔍 Ask Questions About Your Documents")

    
    # Check if API is configured
    if not Config.GEMINI_API_KEY:
        st.warning("⚠️ Please add your Gemini API key in the sidebar to get started")
        
        with st.expander("📘 How to get a free Gemini API key:"):
            st.markdown("""
            1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
            2. Click **"Get API Key"** 
            3. Click **"Create API Key"**
            4. Copy the key and paste it in the sidebar
            """)
        return
    
    # Check if database has documents
    stats = st.session_state.db.get_stats()
    if stats['total_chunks'] == 0:
        st.info("📤 Upload a PDF document to get started!")
        
        # Show example
        with st.expander("🎯 What can I ask?"):
            st.markdown("""
            Once you upload a document, you can ask questions like:
            - "What is the main topic of this document?"
            - "Summarize page 3"
            - "What are the key findings?"
            - "Explain the methodology used"
            """)
    else:
        # Show active documents
        if stats['doc_names']:
            st.caption(f"📄 Active: {', '.join(stats['doc_names'][:3])}")
    
    st.divider()
    
    # ===== Chat Interface =====
    
    # Display conversation history
    for qa in st.session_state.conversation:
        with st.chat_message("user"):
            st.write(qa["question"])
        
        with st.chat_message("assistant"):
            st.write(qa["answer"]["answer"])
            st.caption(f"⚡ {qa['answer']['time']:.1f}s • 📚 {qa['answer']['context_used']} sources • 🤖 Gemini")
            
            if qa.get("context"):
                with st.expander("📖 View Sources"):
                    for i, ctx in enumerate(qa["context"][:3], 1):
                        st.markdown(f"**Source {i}:** {ctx['metadata']['source']} (Page {ctx['metadata']['page']})")
                        st.text(ctx['text'][:200] + "...")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):
                # Search
                results = st.session_state.db.search(prompt)
                
                if not results:
                    answer_text = "No relevant documents found. Please upload a PDF first."
                    st.write(answer_text)
                    answer_meta = {
                        "answer": answer_text,
                        "time": 0,
                        "model": Config.GEMINI_MODEL,
                        "context_used": 0
                    }
                else:
                    # Generate with Gemini
                    answer_meta = GeminiInterface.generate_answer(prompt, results)
                    st.write(answer_meta["answer"])
                    
                    # Show metadata
                    st.caption(f"⚡ {answer_meta['time']:.1f}s • 📚 {answer_meta['context_used']} sources • 🤖 Gemini")
                    
                    # Show sources
                    with st.expander(f"📚 View {len(results)} Sources"):
                        for i, res in enumerate(results[:3], 1):
                            score_pct = res['score'] * 100
                            st.markdown(f"**Source {i}:** {res['metadata']['source']} (Page {res['metadata']['page']})")
                            st.progress(res['score'], text=f"Relevance: {score_pct:.0f}%")
                            st.text(res['text'][:250] + "...")
                
                # Save conversation
                st.session_state.conversation.append({
                    "question": prompt,
                    "answer": answer_meta,
                    "context": results[:3] if results else []
                })

if __name__ == "__main__":
    main()