import React, { useState } from 'react';
import './App.css';
import PDFUploader from './components/PDFUploader';
import ChatInterface from './components/ChatInterface';

function App() {
  const [pdfLoaded, setPdfLoaded] = useState(false);
  const [pdfInfo, setPdfInfo] = useState(null);

  const handlePDFUpload = (info) => {
    setPdfLoaded(true);
    setPdfInfo(info);
  };

  const handleReset = () => {
    setPdfLoaded(false);
    setPdfInfo(null);
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>📄 PDF ChatBot</h1>
        <p>Upload a PDF and ask questions about its content</p>
      </header>

      <div className="app-container">
        {!pdfLoaded ? (
          <PDFUploader onUploadSuccess={handlePDFUpload} />
        ) : (
          <ChatInterface pdfInfo={pdfInfo} onReset={handleReset} />
        )}
      </div>
    </div>
  );
}

export default App;
