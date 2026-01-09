
# RAG Application with ChromaDB and Groq

A Retrieval-Augmented Generation (RAG) application built with Streamlit that allows users to upload documents and ask questions about them using AI.

##  Features

- **Multi-format Support**: Upload PDF, TXT, and Markdown files
- **Intelligent Chunking**: Documents are split into 500-character chunks with 50-character overlap for optimal context preservation
- **Vector Database**: Uses ChromaDB for efficient embedding storage and similarity search
- **Advanced Embeddings**: Powered by sentence-transformers/all-MiniLM-L6-v2 model
- **AI-Powered Answers**: Leverages Groq's llama-3.1-8b-instant model for accurate responses
- **Clean UI**: Simple and intuitive Streamlit interface
- **Context Viewing**: See which document chunks were used to generate answers

##  Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Groq API (llama-3.1-8b-instant)
- **PDF Processing**: PyPDF2
- **Programming Language**: Python 3.8+

##  Prerequisites

- Python 3.8 or higher
- Groq API key (get it from groq.com by creating a account)

##  Dependencies

Create a `requirements.txt` file with:
```
streamlit
sentence-transformers
groq
PyPDF2
chromadb
numpy
torch
```

##  Usage

1. **Run the application**
```bash
streamlit run app.py
```

2. **Access the app**
   - Open your browser and navigate to `http://localhost:8502`

3. **Upload documents**
   - Click "Browse files" and select your documents (.txt, .md, or .pdf)
   - Wait for processing to complete

4. **Ask questions**
   - Type your question in the text input field
   - Click "Get Answer" button
   - View the AI-generated response based on your documents

5. **View context**
   - Expand "View Retrieved Context" to see which chunks were used for the answer

##  Project Structure
```
rag-application/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation

```

### RAG Pipeline

1. **Document Upload**: User uploads documents (PDF/TXT/MD)
2. **Text Extraction**: Extract text from documents
3. **Chunking**: Split text into 500-character chunks with 50-character overlap
4. **Embedding Generation**: Convert chunks to vector embeddings using all-MiniLM-L6-v2
5. **Storage**: Store embeddings in ChromaDB vector database
6. **Query Processing**: Convert user question to embedding
7. **Retrieval**: Find top 3 most similar chunks from ChromaDB
8. **Answer Generation**: Send retrieved context + question to Groq API
9. **Response**: Display AI-generated answer to user





Made with ❤️ using Python and AI
