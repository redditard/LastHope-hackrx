# PDF to Gemini Q&A FastAPI Application

This FastAPI application implements a PDF document question-answering system using Google's Gemini AI. It downloads PDFs from URLs, extracts text, and answers questions about the content.

## 📋 **Key Features**
✅ **Bearer Token Authentication** - Uses the exact token from your API spec  
✅ **PDF URL Processing** - Downloads and extracts text from PDF URLs  
✅ **Gemini AI Integration** - Sends content to Gemini Pro for Q&A  
✅ **Batch Question Processing** - Processes multiple questions efficiently  
✅ **Configurable Batch Size** - Adjust questions per API call via environment variable  
✅ **API Key Rotation** - Automatic failover between multiple API keys  
✅ **Comprehensive Logging** - Detailed logs for debugging and monitoring  
✅ **Error Handling** - Robust error responses and fallback mechanisms  
✅ **FastAPI Documentation** - Auto-generated API docs  

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Optional: Additional API keys for rotation (comma-separated)
# GEMINI_API_KEYS=key1,key2,key3,key4

# Optional: Number of questions to process in each batch (default: 10)
GEMINI_BATCH_SIZE=10

# Key rotation settings
GEMINI_KEY_ROTATION_ENABLED=true
GEMINI_MAX_RETRIES_PER_KEY=2
```

### 3. Get Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

## Running the Application

### Development Mode

```bash
python main.py
```

### Production Mode with Uvicorn

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Usage

### Authentication

All requests to `/hackrx/run` require a Bearer token in the Authorization header:

```
Authorization: Bearer 09988a27dc0bf3ef755e893e2e6650693e4009189215f7824b023cc07db59b1b
```

### Main Endpoint

**POST** `/hackrx/run`

**Request Body:**
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the main topic of this document?",
        "What are the key points mentioned?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "The main topic is...",
        "The key points are..."
    ]
}
```

### Example Request

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
     -H "Authorization: Bearer 09988a27dc0bf3ef755e893e2e6650693e4009189215f7824b023cc07db59b1b" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
       "questions": [
         "What is the grace period for premium payment?",
         "What is the waiting period for pre-existing diseases?"
       ]
     }'
```

## Health Check

**GET** `/health` - Returns API health status

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid PDF URL, extraction failed)
- `401` - Unauthorized (invalid bearer token)
- `422` - Validation Error (invalid request format)
- `500` - Internal Server Error (Gemini API issues)

## Limitations

- **Token Limits**: Large PDFs may exceed Gemini's context window (~8K tokens)
- **Processing Time**: Complex documents may take longer to process
- **No Chunking**: This implementation sends the entire document to Gemini at once

## Production Considerations

For production use, consider implementing:
- Document chunking for large PDFs
- Caching mechanisms
- Rate limiting
- Logging and monitoring
- Vector database for better retrieval
- Queue system for async processing

## Project Structure

```
.
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
├── README.md           # This file
├── intructions.md      # Original implementation guide
└── API spec.md         # API specification
```
