from database import init_db, add_document, get_user_documents, delete_document
from auth import get_current_user
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
from docx import Document
import io
import os
import uuid
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb

load_dotenv()
init_db

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="/app/data/chroma_data")
collection = chroma_client.get_or_create_collection(name="documents")

MAX_FILE_SIZE_MB = 5

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    if filename.endswith('.pdf'):
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    elif filename.endswith('.docx'):
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file type")

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.pdf', '.docx']:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")
    try:
        text = extract_text_from_file(contents, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    text = "\n".join([line for line in text.splitlines() if line.strip()])
    return {"filename": file.filename, "text": text[:1000] + "..." if len(text) > 1000 else text}

@app.post("/embed")
async def embed_document(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.pdf', '.docx']:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are allowed")

    try:
        text = extract_text_from_file(contents, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found")

    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )
    embeddings = [data.embedding for data in response.data]

    doc_id = str(uuid.uuid4())
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i]],
            ids=[f"{doc_id}_{i}"],
            metadatas=[{"filename": file.filename, "chunk_index": i, "doc_id": doc_id, "user_id": user_id}]
        )
    # Insert the new line here
    add_document(doc_id, user_id, file.filename, len(chunks))
    return {
        "status": "success",
        "chunks_embedded": len(chunks),
        "doc_id": doc_id,
        "filename": file.filename
    }
from pydantic import BaseModel

# We'll use a Pydantic model to validate the incoming JSON
class AskRequest(BaseModel):
    question: str
    doc_id: str | None = None   # optional; if omitted we search all documents

@app.post("/ask")
async def ask_question(request: AskRequest, user_id: str = Depends(get_current_user)):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # 1. Embed the question
    q_embedding = client.embeddings.create(
        input=[question],
        model="text-embedding-3-small"
    ).data[0].embedding

    # 2. Retrieve the top 4 most relevant chunks
    # Build filter – ChromaDB requires $and for multiple conditions
    conditions = [{"user_id": user_id}]
    if request.doc_id:
        conditions.append({"doc_id": request.doc_id})
    where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=8,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )

    # Extract the text of the retrieved chunks
    retrieved_chunks = results["documents"][0]  # list of strings
    if not retrieved_chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found. Upload a document first.")

    # 3. Build a prompt with context
    context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])

    prompt = f"""You are a helpful assistant that answers questions **only** based on the provided document chunks.
Try to give a complete answer based on the chunks, even if the information is scattered across them.
If the chunks are entirely irrelevant or contain no information related to the question, only then say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    # 4. Call GPT-4o to generate the answer
    chat_response = client.chat.completions.create(
        model="gpt-4o",   # you can also use gpt-4o-mini for lower cost
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,   # low creativity, more factual
        max_tokens=500
    )

    answer = chat_response.choices[0].message.content

    # 5. Return answer with source citations
    return {
        "answer": answer,
        "sources": [
            {"text": chunk, "metadata": meta}
            for chunk, meta in zip(retrieved_chunks, results["metadatas"][0])
        ]
    }