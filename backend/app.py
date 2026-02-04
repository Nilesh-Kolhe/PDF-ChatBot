import os
import json
import asyncio
import tempfile
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.callbacks.base import AsyncCallbackHandler
from typing import Any, Dict, List

load_dotenv()

app = FastAPI(title="PDF Q&A APP - Fully Local Version")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
vector_store = None
qa_chain = None
chat_history = []

# Config
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama-local-key")

# Models
class AskRequest(BaseModel):
    question: str

def allowed_file(filename):
    """Check If File is a PDF"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(file_path: str) -> int:
    """Process PDF and Create QA Chain with Local Models"""
    global vector_store, qa_chain, chat_history

    print("\n" + "=" * 60)
    print("📄 Processing PDF...")
    print("=" * 60)

    # Load PDF
    print("\n1️⃣ Loading PDF Document...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")

    # Split into chunks
    print("\n2️⃣ Splitting Document into Chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} Chunks")

    # DEBUG: Print each chunk
    print("\n📦 Chunk Preview:")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk.page_content[:1400].replace("\n", " ")
        print(f"Chunk {i} (Page {chunk.metadata.get('page', 'N/A')}): {preview}...")

    # Create Embeddings using Local Model (No API Needed!)
    print("\n3️⃣ Loading Local Embedding Model...")
    print("📦 Model: all-MiniLM-L6-v2 (Runs on Your Mac)")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    print("✅ Embedding Model Loaded")

    # Create vector store
    print("\n4️⃣ Creating Vector Database...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector Store Created")

    # Setup memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    # Initialize Ollama LLM (Qwen 2.5 14B - Running Locally !)
    print("\n5️⃣ Connecting to Ollama (Qwen 2.5 14B)...")
    print("🤖 Model: qwen2.5:14b (local, no API needed)")
    try:
        llm = ChatOpenAI(
            model=OLLAMA_MODEL_NAME,
            temperature=0.1,
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            streaming=True,
        )
        print("✅ Connected to Ollama successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not Connect to Ollama: {e}")
        print("💡 Make Sure Ollama is Running: 'brew services start ollama'")
        raise Exception(
            "Ollama Service not Running. Please Start it With: brew services start ollama"
        )

    # Create QA chain
    print("\n6️⃣ Creating Conversational QA Chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 12},
        ),
        memory=memory,
        return_source_documents=True,
        verbose=True,
    )
    print("✅ QA Chain Ready!")

    chat_history = []

    print("\n" + "=" * 60)
    print("✅ PDF PROCESSING COMPLETE!")
    print("=" * 60)
    print("\n🚀 Ready to Answer Questions!\n")

    return len(chunks)

@app.get("/api/ask-stream-test")
async def ask_question_stream():
    async def event_generator():
        for i in range(5):
            payload = {"token": f"tick-{i} "}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(1)
        # final message
        final_payload = {"final": "Done!", "sources": []}
        yield f"data: {json.dumps(final_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/ask-stream")
async def ask_question_stream(request: Request, question: str):
    """Stream Answer via SSE."""
    global qa_chain

    if qa_chain is None:
        raise HTTPException(status_code=400, detail="Please Upload a PDF First")

    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is Required")

    # Queue Where Callback Pushes Tokens and Final Message
    queue: asyncio.Queue = asyncio.Queue()

    class StreamHandler(AsyncCallbackHandler):
        """LangChain Async Callback to Push Tokens into the Queue."""

        def __init__(self, queue: asyncio.Queue) -> None:
            super().__init__()
            self.queue = queue
            self._final_text_parts: List[str] = []

        # --- Required Stubs so LangChain doesn't Raise NotImplementedError ---

        async def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[Dict[str, Any]]],
            **kwargs: Any,
        ) -> None:
            # No-op
            return

        async def on_chat_model_end(self, response, **kwargs: Any) -> None:
            return

        async def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            **kwargs: Any,
        ) -> None:
            return

        async def on_llm_end(self, response, **kwargs: Any) -> None:
            # Some models call LLM-level hooks instead of chat_model ones
            return

        # --- Actual Streaming Hook used for Tokens ---
        async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            self._final_text_parts.append(token)
            await self.queue.put({"token": token})

    async def run_chain():
        """Run qa_chain with Streaming Callbacks and Push Final Message."""
        handler = StreamHandler(queue)

        # If qa_chain supports async:
        # response = await qa_chain.acall({"question": question}, callbacks=[handler])

        # If qa_chain is Sync but Accepts Callbacks, Run in a Thread:
        response = await asyncio.to_thread(
            qa_chain,
            {"question": question},
            callbacks=[handler],  # adapt if your chain passes callbacks differently
        )

        # Build Final Answer and Sources from the Chain Response
        try:
            answer = response["answer"]
            # --- Strip 'Standalone Question:' Noise If Present ---
            for marker in ["Standalone Question:", "Standalone question:"]:
                if marker in answer:
                    answer = answer.split(marker, 1)[-1].strip()
            # -----------------------------------------------------
            source_docs = response.get("source_documents", [])
            sources = [
                {
                    "content": doc.page_content[:50] + "...",
                    "page": doc.metadata.get("page", "N/A"),
                }
                for doc in source_docs
            ]
            await queue.put({"final": answer, "sources": sources})
        except Exception as e:
            await queue.put({"error": f"Failed to Build Final Answer: {str(e)}"})

        # Sentinel to Close the SSE Stream
        await queue.put(None)

    # The Generator
    async def event_generator():
        task = asyncio.create_task(run_chain())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            task.cancel()
    # Return type to be StreamingResponse for Stream
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF Upload"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No File Selected")

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Only PDF Files are Allowed")

    try:
        # Save to a Temp Directory
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        num_chunks = process_pdf(file_path)

        # Cleanup
        os.remove(file_path)
        os.rmdir(temp_dir)

        return {
            "message": "PDF Uploaded and Processed Successfully",
            "filename": file.filename,
            "chunks": num_chunks,
        }
    except Exception as e:
        print(f"\nError Uploading PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
async def ask_question(payload: AskRequest):
    """Handle question answering"""
    global qa_chain

    if qa_chain is None:
        raise HTTPException(status_code=400, detail="Please Upload a PDF First")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is Required")

    try:
        print(f"\n💬 Question: {question}")
        print("🤔 Thinking...")

        response = qa_chain({"question": question})
        answer = response["answer"]
        source_docs = response.get("source_documents", [])

        print(f"✅ Answer: {answer[:200]}...")

        sources = []
        print("\n" + "=" * 60)
        print("🔍 DEBUG: Retrieved Chunks")
        print("=" * 60)
        for i, doc in enumerate(source_docs, 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Content: {doc.page_content[:len(doc.page_content)]}...")
            print(f"Full length: {len(doc.page_content)} chars")
            sources.append(
                {
                    "content": doc.page_content[:50] + "...",
                    "page": doc.metadata.get("page", "N/A"),
                }
            )
        print("=" * 60 + "\n")

        return {"answer": answer, "sources": sources}
    except Exception as e:
        print(f"\n❌ Error Answering Question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset_session():
    """Reset the Session"""

    global vector_store, qa_chain, chat_history
    vector_store = None
    qa_chain = None
    chat_history = []

    print("\nSession Reset")
    return {'message': 'Session Reset Successfully'}

@app.get("/api/health")
async def health_check():
    """Health Check Endpoint"""
    # Check if Ollama is Running
    ollama_status = "Unknown"
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        ollama_status = "Running" if response.status_code == 200 else "Error"
    except Exception:
        ollama_status = "Not Running"

    return {
        'status': 'Healthy',
        'pdf_loaded': qa_chain is not None,
        'ollama_status': ollama_status,
        'model': 'qwen2.5:14b',
        'embedding_model': 'all-MiniLM-L6-v2 (local)'
    }