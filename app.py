import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import PyPDF2
import chromadb
from chromadb.config import Settings
import uuid

# CONFIGURE YOUR GROQ API KEY HERE
GROQ_API_KEY = "add your groq api key here"

# Page configuration
st.set_page_config(page_title="RAG Application", page_icon="📚", layout="centered")

# Title
st.title("RAG Application")
st.write("Powered by Groq, ChromaDB & all-MiniLM-L6-v2")
st.markdown("---")

# Initialize session state
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = chromadb.EphemeralClient()
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model"""
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
        if end == len(text):
            break
    
    return chunks

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def create_or_get_collection():
    """Create or get ChromaDB collection"""
    try:
        # Try to delete existing collection
        try:
            st.session_state.chroma_client.delete_collection("documents")
        except:
            pass
        
        # Create new collection
        collection = st.session_state.chroma_client.create_collection(
            name="documents"
        )
        return collection
    except Exception as e:
        st.error(f"Error creating collection: {str(e)}")
        return None

def process_document(file, model):
    """Process uploaded document and add to ChromaDB"""
    try:
        # Extract text based on file type
        if file.type == "application/pdf":
            text = extract_text_from_pdf(file)
        else:
            text = file.read().decode('utf-8')
        
        if not text.strip():
            st.warning(f"No text found in {file.name}")
            return None
        
        # Chunk the text
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        
        # Generate embeddings using the model
        embeddings = model.encode(chunks).tolist()
        
        # Add to ChromaDB with embeddings
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": file.name, "chunk_index": i} for i in range(len(chunks))]
        
        st.session_state.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            'name': file.name,
            'num_chunks': len(chunks)
        }
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None

def retrieve_relevant_chunks(query, model, top_k=3):
    """Retrieve most relevant chunks from ChromaDB using embeddings"""
    try:
        # Generate query embedding using the model
        query_embedding = model.encode([query])[0].tolist()
        
        # Query ChromaDB with the embedding
        results = st.session_state.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        st.error(f"Error retrieving chunks: {str(e)}")
        return []

def generate_answer(query, context):
    """Generate answer using Groq API"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1000,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# File upload section
st.subheader("Upload Documents")

uploaded_files = st.file_uploader(
    "Choose files to upload",
    type=['txt', 'md', 'pdf'],
    accept_multiple_files=True,
    help="Upload .txt, .md, or .pdf files"
)

if uploaded_files:
    # Load model if not already loaded
    if st.session_state.model is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.model = load_embedding_model()
    
    # Create collection if not exists
    if st.session_state.collection is None:
        st.session_state.collection = create_or_get_collection()
    
    if st.session_state.model and st.session_state.collection:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        
        if new_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(new_files):
                status_text.text(f"Processing: {file.name}...")
                
                result = process_document(file, st.session_state.model)
                
                if result:
                    st.session_state.documents.append(result)
                    st.session_state.processed_files.add(file.name)
                
                progress_bar.progress((idx + 1) / len(new_files))
            
            status_text.success("Processing complete!")
            progress_bar.empty()

# Display processed documents
if st.session_state.documents:
    st.success(f"{len(st.session_state.documents)} document(s) loaded in ChromaDB")
    
    with st.expander("View Document Details"):
        total_chunks = 0
        for doc in st.session_state.documents:
            st.write(f"- {doc['name']} ({doc['num_chunks']} chunks)")
            total_chunks += doc['num_chunks']
        st.info(f"Total chunks in database: {total_chunks} (500 chars each, 50 char overlap)")
    
    if st.button("Clear All Documents"):
        st.session_state.documents = []
        st.session_state.processed_files = set()
        st.session_state.collection = create_or_get_collection()
        st.rerun()

st.markdown("---")

# Query section
st.subheader("Ask Questions")

if len(st.session_state.documents) > 0:
    query = st.text_input(
        "Enter your question:",
        placeholder="What would you like to know about your documents?"
    )
    
    if st.button("Get Answer", type="primary", use_container_width=True):
        if query.strip():
            with st.spinner("Generating answer..."):
                # Load model if not loaded
                if st.session_state.model is None:
                    st.session_state.model = load_embedding_model()
                
                if st.session_state.model:
                    # Retrieve relevant chunks from ChromaDB using embeddings
                    relevant_chunks = retrieve_relevant_chunks(
                        query,
                        st.session_state.model,
                        top_k=3
                    )
                    
                    if relevant_chunks:
                        # Generate answer using Groq
                        context = "\n\n".join(relevant_chunks)
                        answer = generate_answer(query, context)
                        
                        # Display answer
                        st.markdown("### Response:")
                        st.write(answer)
                        
                        # Show retrieved context
                        with st.expander("View Retrieved Context"):
                            for i, chunk in enumerate(relevant_chunks, 1):
                                st.markdown(f"**Chunk {i}:**")
                                st.text(chunk)
                                if i < len(relevant_chunks):
                                    st.markdown("---")
                    else:
                        st.error("Could not retrieve relevant information")
        else:
            st.warning("Please enter a question")
        
else:
    st.info("Please upload documents first to start asking questions")

# Footer
st.markdown("---")
st.caption("Built with Streamlit, ChromaDB & Groq")