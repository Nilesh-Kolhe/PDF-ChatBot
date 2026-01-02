import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)

# Enable CORS for all routes with permissive settings
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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

vector_store = None
qa_chain = None
chat_history = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_pdf(file_path):
    global vector_store, qa_chain, chat_history
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        verbose=True
    )
    
    chat_history = []
    return len(chunks)

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_pdf():
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
        
        os.remove(file_path)
        os.rmdir(temp_dir)
        
        return jsonify({
            'message': 'PDF uploaded and processed successfully',
            'filename': filename,
            'chunks': num_chunks
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask', methods=['POST', 'OPTIONS'])
def ask_question():
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
        response = qa_chain({"question": question})
        answer = response['answer']
        source_docs = response.get('source_documents', [])
        
        sources = []
        for doc in source_docs:
            sources.append({
                'content': doc.page_content[:200] + '...',
                'page': doc.metadata.get('page', 'N/A')
            })
        
        return jsonify({
            'answer': answer,
            'sources': sources
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST', 'OPTIONS'])
def reset_session():
    if request.method == 'OPTIONS':
        return '', 204
        
    global vector_store, qa_chain, chat_history
    vector_store = None
    qa_chain = None
    chat_history = []
    return jsonify({'message': 'Session reset successfully'}), 200

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return '', 204
        
    return jsonify({
        'status': 'healthy',
        'pdf_loaded': qa_chain is not None
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')