# PDF ChatBot Backend

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   - Mac/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`

3. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

4. Create `.env` file:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and add your OpenAI API key if used.

5. Run the server:
   ```bash
   python app.py
   ```

Backend runs on http://localhost:5000

## API Endpoints

- POST /api/upload - Upload PDF file
- POST /api/ask - Ask question about PDF
- POST /api/reset - Reset session
- GET /api/health - Health check
- GET /api/ask-stream - Stream the response
- GET /api/ask-stream-test - Test response stream
