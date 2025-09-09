import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

app = FastAPI(title="PDF Semantic Search API")

# Configure CORS - THIS IS CRITICAL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - change in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Store vector stores in memory (in production, use a persistent storage)
vector_stores = {}

# Create uploads directory in parent directory to avoid reload triggers
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class SearchRequest(BaseModel):
    query: str
    pdf_id: str

class PDFUploadResponse(BaseModel):
    pdf_id: str
    page_count: int
    chunk_count: int

class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: float

@app.post("/upload-pdf/", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Generate unique ID for this PDF
    pdf_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{pdf_id}.pdf")
    
    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Load the PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Split the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Create vector store
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents=all_splits)
        
        # Store the vector store
        vector_stores[pdf_id] = vector_store
        
        return PDFUploadResponse(
            pdf_id=pdf_id,
            page_count=len(docs),
            chunk_count=len(all_splits)
        )
        
    except Exception as e:
        # Clean up the uploaded file if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/search/", response_model=List[SearchResult])
async def search_pdf(request: SearchRequest):
    if request.pdf_id not in vector_stores:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    try:
        vector_store = vector_stores[request.pdf_id]
        results = vector_store.similarity_search_with_score(request.query, k=1)
        
        search_results = []
        for doc, score in results:
            search_results.append(SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score
            ))
        
        return search_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching PDF: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "uploads_dir": UPLOAD_DIR}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)