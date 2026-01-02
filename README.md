# PDF ChatBot

A full-stack application that allows users to upload PDF files and ask questions about their content using AI.

## 📋 Features

- 📄 **PDF Upload**: Drag-and-drop interface for easy PDF uploads
- 🤖 **AI-Powered Q&A**: Ask questions about your PDF content and get intelligent answers
- 🔍 **Source Citations**: See relevant excerpts and page numbers from the PDF
- 💬 **Chat Interface**: Interactive conversation with context awareness
- 🎨 **Modern UI**: Beautiful, responsive design with smooth animations

## 🛠 Tech Stack

### Backend
- Flask (Python web framework)
- LangChain (LLM framework)
- Ollama qwen2.5:14b
- FAISS (Vector database)
- PyPDF for PDF processing

### Frontend
- React 18
- Axios for API calls
- react-dropzone for file uploads
- react-icons for UI icons

## 📦 Prerequisites
- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn
- No Key if running a local model. OpenAI API key ([Get one here](https://platform.openai.com/api-keys)).

## 🚀 Installation & Setup

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=your_key_here

# Run the backend server
python app.py
```

Backend will run on **http://localhost:5000**

### 2. Frontend Setup (Open a new terminal)

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

Frontend will automatically open at **http://localhost:3000**

## 💡 Usage

1. **Upload a PDF**: Drag and drop a PDF file or click to browse
2. **Wait for Processing**: The app will process and chunk your PDF (takes a few seconds)
3. **Ask Questions**: Type your questions in the chat interface
4. **View Answers**: Get AI-generated answers with source citations showing page numbers
5. **Upload New PDF**: Click "Upload New PDF" button to start over with a different document

## 📁 Project Structure

```
pdf-qa-app/
├── backend/
│   ├── app.py              # Flask application with API endpoints
│   ├── requirements.txt    # Python dependencies
│   ├── .env.example        # Environment variables template
│   └── README.md          # Backend documentation
├── frontend/
│   ├── public/            # Static files
│   │   ├── index.html
│   │   └── robots.txt
│   ├── src/
│   │   ├── components/    # React components
│   │   │   ├── PDFUploader.js      # PDF upload component
│   │   │   ├── PDFUploader.css
│   │   │   ├── ChatInterface.js    # Chat UI component
│   │   │   └── ChatInterface.css
│   │   ├── App.js         # Main App component
│   │   ├── App.css        # Global styles
│   │   ├── index.js       # React entry point
│   │   └── index.css      # Base styles
│   ├── package.json       # Node dependencies
│   └── README.md         # Frontend documentation
└──.gitignore
└── README.md             # This file
```

## 🔧 Configuration Options

### Chunk Size (in backend/app.py)
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # Adjust this for larger/smaller chunks
    chunk_overlap=400,    # Overlap between chunks
)
```

### Model Selection (in backend/app.py)
```python
llm = OllamaLLM(
    model="qwen2.5:14b",
    temperature=0.1, # Initially 0.7
    base_url="http://localhost:11434",
    verbose=True
)
```

### Max File Size (in backend/app.py)
```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB default
```

## 🐛 Troubleshooting

### Backend Issues

**Problem**: `ModuleNotFoundError`
**Solution**: Ensure virtual environment is activated and run `pip install -r requirements.txt`

**Problem**: `OpenAI API Error`
**Solution**: Check that your API key is correctly set in the `.env` file

**Problem**: `Port 5000 already in use`
**Solution**: Change the port in `app.py` (last line) and update the frontend API URL in component files

### Frontend Issues

**Problem**: `Connection refused to localhost:5000`
**Solution**: Make sure the backend server is running before starting the frontend

**Problem**: `Module not found`
**Solution**: Run `npm install` in the frontend directory

**Problem**: `PDF upload fails`
**Solution**: Check file size (max 16MB) and ensure it's a valid PDF file

## 🚀 Deployment

### Backend
- Deploy to Heroku, AWS, or DigitalOcean
- Set environment variables in your hosting platform
- Consider using Pinecone for production vector database

### Frontend
- Build: `npm run build`
- Deploy to Vercel, Netlify, or AWS S3
- Update API endpoint URL from localhost to your backend URL

## 📄 License

MIT License - Feel free to use this project for learning and development.

## 🙏 Credits

Built with ❤️ using:
- React (https://react.dev/)
- Flask (https://flask.palletsprojects.com/)
- LangChain (https://python.langchain.com/)
- OpenAI (https://openai.com/)

## 📞 Support

For issues or questions:
1. Check the README files in backend/ and frontend/ directories
2. Review the troubleshooting section above
3. Check that all dependencies are properly installed
4. Ensure your OpenAI API key is valid and has credits

---

**Happy Coding! 🎉**
