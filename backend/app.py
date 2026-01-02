import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)

# Enable CORS for frontend
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
vector_store = None
qa_chain = None
chat_history = []

def allowed_file(filename):
    """Check if file is a PDF"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(file_path):
    """Process PDF and create QA chain with local models"""
    global vector_store, qa_chain, chat_history

    print("\n" + "="*60)
    print("📄 Processing PDF...")
    print("="*60)

    # Load PDF
    print("\n1️⃣  Loading PDF document...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")

    # Split into chunks
    print("\n2️⃣  Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, # Initially 1000
        chunk_overlap=400, # Initially 200
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]  # Split at natural boundaries
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")

    # DEBUG: Print each chunk
    print("\n📦 Chunk Preview:")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk.page_content[:1400].replace('\n', ' ')
        print(f"Chunk {i} (Page {chunk.metadata.get('page', 'N/A')}): {preview}...")

    # Create embeddings using local model (no API needed!)
    print("\n3️⃣  Loading local embedding model...")
    print("   📦 Model: all-MiniLM-L6-v2 (runs on your Mac)")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    print("   ✅ Embedding model loaded")

    # Create vector store
    print("\n4️⃣  Creating vector database...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("   ✅ Vector store created")

    # Setup memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    # Initialize Ollama LLM (Qwen 2.5 14B - running locally!)
    print("\n5️⃣  Connecting to Ollama (Qwen 2.5 14B)...")
    print("  🤖 Model: qwen2.5:14b (local, no API needed)")
    try:
        llm = OllamaLLM(
            model="qwen2.5:14b",
            temperature=0.1, # Initially 0.7
            base_url="http://localhost:11434",
            verbose=True
        )
        print("   ✅ Connected to Ollama successfully")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not connect to Ollama: {e}")
        print("   💡 Make sure Ollama is running: 'brew services start ollama'")
        raise Exception("Ollama service not running. Please start it with: brew services start ollama")

    # Create QA chain
    print("\n6️⃣  Creating conversational QA chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 12, # Intially 4
            }),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    print("✅ QA chain ready!")

    chat_history = []

    print("\n" + "="*60)
    print("✅ PDF PROCESSING COMPLETE!")
    print("="*60)
    print("\n🚀 Ready to answer questions!\n")

    return len(chunks)

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_pdf():
    """Handle PDF upload"""
    if request.method == 'OPTIONS':
        return '', 204

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    try:
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        file.save(file_path)

        num_chunks = process_pdf(file_path)

        # Cleanup
        os.remove(file_path)
        os.rmdir(temp_dir)

        return jsonify({
            'message': 'PDF uploaded and processed successfully',
            'filename': filename,
            'chunks': num_chunks
        }), 200

    except Exception as e:
        print(f"\n❌ Error uploading PDF: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask', methods=['POST', 'OPTIONS'])
def ask_question():
    """Handle question answering"""
    if request.method == 'OPTIONS':
        return '', 204

    global qa_chain

    if qa_chain is None:
        return jsonify({'error': 'Please upload a PDF first'}), 400

    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'Question is required'}), 400

    try:
        print(f"\n💬 Question: {question}")
        print("🤔 Thinking...")

        response = qa_chain({"question": question})
        answer = response['answer']
        source_docs = response.get('source_documents', [])

        print(f"✅ Answer: {answer[:200]}...")

        # ADD THIS DEBUG SECTION:
        sources = []
        print("\n" + "="*60)
        print("🔍 DEBUG: Retrieved Chunks")
        print("="*60)
        for i, doc in enumerate(source_docs, 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Content: {doc.page_content[:len(doc.page_content)]}...")  # First 300 chars
            print(f"Full length: {len(doc.page_content)} chars")
            sources.append({
                'content': doc.page_content[:50] + '...',
                'page': doc.metadata.get('page', 'N/A')
            })
        print("="*60 + "\n")

        # sources = []
        # for doc in source_docs:
        #     sources.append({
        #         'content': doc.page_content[:50] + '...',
        #         'page': doc.metadata.get('page', 'N/A')
        #     })

        return jsonify({
            'answer': answer,
            'sources': sources
        }), 200

    except Exception as e:
        print(f"\n❌ Error answering question: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST', 'OPTIONS'])
def reset_session():
    """Reset the session"""
    if request.method == 'OPTIONS':
        return '', 204

    global vector_store, qa_chain, chat_history
    vector_store = None
    qa_chain = None
    chat_history = []

    print("\n🔄 Session reset")
    return jsonify({'message': 'Session reset successfully'}), 200

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return '', 204

    # Check if Ollama is running
    ollama_status = "unknown"
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        ollama_status = "running" if response.status_code == 200 else "error"
    except:
        ollama_status = "not running"

    return jsonify({
        'status': 'healthy',
        'pdf_loaded': qa_chain is not None,
        'ollama_status': ollama_status,
        'model': 'qwen2.5:14b',
        'embedding_model': 'all-MiniLM-L6-v2 (local)'
    }), 200

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 PDF Q&A APP - FULLY LOCAL VERSION")
    print("="*60)
    print("\n📦 Using:")
    print("   • LLM: Qwen 2.5 14B (via Ollama)")
    print("   • Embeddings: all-MiniLM-L6-v2 (local)")
    print("   • Vector Store: FAISS")
    print("\n✅ No API keys needed!")
    print("✅ No quota limits!")
    print("✅ Completely offline!")
    print("\n💡 Make sure Ollama is running:")
    print("   brew services start ollama")
    print("   ollama pull qwen2.5:14b")
    print("\n" + "="*60)
    print("🌐 Starting server on http://localhost:5000")
    print("="*60 + "\n")

    app.run(debug=True, port=5000, host='0.0.0.0')