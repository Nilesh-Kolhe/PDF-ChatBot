import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { FiSend, FiRefreshCw } from 'react-icons/fi';
import './ChatInterface.css';

const ChatInterface = ({ pdfInfo, onReset }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    setMessages([
      {
        type: 'system',
        content: `PDF "${pdfInfo.filename}" Loaded Successfully! (${pdfInfo.chunks} Chunks). You can Now Ask Questions About the Content.`
      }
    ]);
  }, [pdfInfo]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!input.trim() || loading) return;

    const userMessage = { type: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:5000/api/ask', {
        question: input
      });

      const aiMessage = {
        type: 'ai',
        content: response.data.answer,
        sources: response.data.sources
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (err) {
      const errorMessage = {
        type: 'error',
        content: err.response?.data?.error || 'Failed to get answer'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    try {
      await axios.post('http://localhost:5000/api/reset');
      onReset();
    } catch (err) {
      console.error('Reset failed:', err);
    }
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <div className="pdf-info">
          <h3>📄 {pdfInfo.filename}</h3>
          <span>{pdfInfo.chunks} chunks processed</span>
        </div>
        <button onClick={handleReset} className="reset-btn">
          <FiRefreshCw /> Upload New PDF
        </button>
      </div>

      <div className="messages-container">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.type}`}>
            <div className="message-content">
              {msg.content}
            </div>
            {msg.sources && msg.sources.length > 0 && (
              <div className="sources">
                <strong>Sources:</strong>
                {msg.sources.map((source, i) => (
                  <div key={i} className="source-item">
                    <small>Page {source.page}: {source.content}</small>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="message ai">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about the PDF..."
          disabled={loading}
          className="question-input"
        />
        <button type="submit" disabled={loading || !input.trim()} className="send-btn">
          <FiSend />
        </button>
      </form>
    </div>
  );
};

export default ChatInterfaceOld;
