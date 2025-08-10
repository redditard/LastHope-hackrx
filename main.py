from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List
import requests
import json
import pdfplumber
import tempfile
import os
import logging
import threading
from io import BytesIO
from config import settings

# Validate configuration on startup
settings.validate()

class GeminiKeyManager:
    """Manages API key rotation with error handling and failover."""
    
    def __init__(self, logger_instance):
        self.logger = logger_instance
        self.api_keys = settings.GEMINI_API_KEYS.copy()
        self.current_key_index = 0
        self.key_retry_counts = {key: 0 for key in self.api_keys}
        self.lock = threading.Lock()
        self.max_retries_per_key = settings.GEMINI_MAX_RETRIES_PER_KEY
        self.rotation_enabled = settings.GEMINI_KEY_ROTATION_ENABLED
        
        self.logger.info(f"GeminiKeyManager initialized with {len(self.api_keys)} key(s)")
        self.logger.info(f"Key rotation enabled: {self.rotation_enabled}")
        self.logger.info(f"Max retries per key: {self.max_retries_per_key}")
    
    def get_current_key(self) -> str:
        """Get the current API key."""
        with self.lock:
            if not self.api_keys:
                raise ValueError("No API keys available")
            return self.api_keys[self.current_key_index]
    
    def get_current_key_info(self) -> dict:
        """Get information about the current key."""
        with self.lock:
            current_key = self.api_keys[self.current_key_index]
            return {
                "key_index": self.current_key_index,
                "key_preview": f"{current_key[:10]}...{current_key[-4:]}",
                "retry_count": self.key_retry_counts[current_key],
                "total_keys": len(self.api_keys)
            }
    
    def mark_key_failed(self, api_key: str) -> bool:
        """Mark a key as failed and optionally rotate to next key.
        
        Returns:
            bool: True if rotated to a new key, False if no more keys available
        """
        if not self.rotation_enabled:
            self.logger.warning("Key rotation disabled, not rotating despite failure")
            return False
            
        with self.lock:
            if api_key in self.key_retry_counts:
                self.key_retry_counts[api_key] += 1
                self.logger.warning(f"Key failure recorded for {api_key[:10]}... (attempt {self.key_retry_counts[api_key]})")
                
                # Check if we should rotate to next key
                if self.key_retry_counts[api_key] >= self.max_retries_per_key:
                    return self._rotate_to_next_key()
            
            return False
    
    def _rotate_to_next_key(self) -> bool:
        """Rotate to the next available key.
        
        Returns:
            bool: True if rotated successfully, False if no more keys available
        """
        original_index = self.current_key_index
        
        # Try to find a key that hasn't exceeded retry limit
        for _ in range(len(self.api_keys)):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            new_key = self.api_keys[self.current_key_index]
            
            if self.key_retry_counts[new_key] < self.max_retries_per_key:
                self.logger.info(f"Rotated to key {self.current_key_index + 1}/{len(self.api_keys)} ({new_key[:10]}...)")
                return True
        
        # If we get here, all keys have exceeded retry limit
        self.current_key_index = original_index
        self.logger.error("All API keys have exceeded retry limit!")
        return False
    
    def reset_key_failures(self, api_key: str = None):
        """Reset failure count for a specific key or all keys."""
        with self.lock:
            if api_key:
                if api_key in self.key_retry_counts:
                    self.key_retry_counts[api_key] = 0
                    self.logger.info(f"Reset failure count for key {api_key[:10]}...")
            else:
                # Reset all keys
                self.key_retry_counts = {key: 0 for key in self.api_keys}
                self.logger.info("Reset failure counts for all keys")
    
    def get_stats(self) -> dict:
        """Get statistics about key usage and failures."""
        with self.lock:
            return {
                "total_keys": len(self.api_keys),
                "current_key_index": self.current_key_index,
                "rotation_enabled": self.rotation_enabled,
                "max_retries_per_key": self.max_retries_per_key,
                "key_failure_counts": {
                    f"{key[:10]}...": count 
                    for key, count in self.key_retry_counts.items()
                }
            }

# Configure logging
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("=== PDF to Gemini Q&A API Starting ===")
logger.info(f"Log level: {settings.LOG_LEVEL}")
logger.info(f"Gemini model: {settings.GEMINI_BASE_URL.split('/')[-1].split(':')[0]}")
logger.info(f"Batch size: {settings.GEMINI_BATCH_SIZE} questions per batch")
logger.info(f"API version: {settings.API_VERSION}")
logger.info("=========================================")

# Initialize the key manager after logger is set up
key_manager = GeminiKeyManager(logger)

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

# Security
security = HTTPBearer()

# Pydantic models
class QueryRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Authentication dependency
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.VALID_BEARER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def extract_pdf_text_from_url(pdf_url: str) -> str:
    """Download PDF from URL and extract text."""
    logger.info(f"Starting PDF text extraction from URL: {pdf_url}")
    
    try:
        # Download the PDF
        logger.info("Downloading PDF...")
        response = requests.get(pdf_url, timeout=settings.PDF_DOWNLOAD_TIMEOUT)
        logger.info(f"PDF download status: {response.status_code}")
        logger.debug(f"PDF content length: {len(response.content)} bytes")
        
        response.raise_for_status()
        
        # Extract text using pdfplumber
        logger.info("Extracting text from PDF...")
        text = ""
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"PDF contains {total_pages} pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    logger.debug(f"Page {page_num}: extracted {len(page_text)} characters")
                else:
                    logger.warning(f"Page {page_num}: no text extracted")
        
        extracted_length = len(text.strip())
        logger.info(f"Total text extracted: {extracted_length} characters")
        
        if extracted_length == 0:
            logger.error("No text could be extracted from PDF")
            raise ValueError("PDF appears to be empty or contains no extractable text")
        
        return text.strip()
    
    except requests.exceptions.Timeout as e:
        logger.error(f"PDF download timeout: {e}")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="PDF download timed out - file may be too large"
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(f"PDF download connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to PDF URL - check URL validity"
        )
    except requests.exceptions.HTTPError as e:
        logger.error(f"PDF download HTTP error: {e}")
        if response.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="PDF not found at the provided URL"
            )
        elif response.status_code == 403:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to PDF URL"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error downloading PDF: HTTP {response.status_code}"
            )
    except requests.RequestException as e:
        logger.error(f"PDF download request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error downloading PDF: {str(e)}"
        )
    except Exception as e:
        logger.error(f"PDF text extraction error: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error extracting text from PDF: {str(e)}"
        )

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def process_questions_in_batches(content: str, questions: List[str]) -> List[str]:
    """Process questions in configurable batches to manage API limits."""
    batch_size = settings.GEMINI_BATCH_SIZE
    total_questions = len(questions)
    
    logger.info(f"Processing {total_questions} questions in batches of {batch_size}")
    
    if total_questions <= batch_size:
        # If we have fewer questions than batch size, process all at once
        logger.info("Questions fit in single batch, processing all together")
        return query_gemini_batch(content, questions)
    
    # Split questions into batches
    question_batches = chunk_list(questions, batch_size)
    all_answers = []
    
    logger.info(f"Split into {len(question_batches)} batches")
    
    for batch_num, question_batch in enumerate(question_batches, 1):
        logger.info(f"Processing batch {batch_num}/{len(question_batches)} with {len(question_batch)} questions")
        
        try:
            batch_answers = query_gemini_batch(content, question_batch)
            all_answers.extend(batch_answers)
            logger.info(f"Batch {batch_num} completed successfully, got {len(batch_answers)} answers")
            
        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}")
            # Add error placeholders for this batch
            error_message = f"Error processing batch {batch_num}: {str(e)}"
            batch_errors = [error_message] * len(question_batch)
            all_answers.extend(batch_errors)
            logger.warning(f"Added {len(batch_errors)} error placeholders for failed batch {batch_num}")
    
    logger.info(f"Batch processing complete: {len(all_answers)} total answers for {total_questions} questions")
    
    # Ensure we have the right number of answers
    if len(all_answers) != total_questions:
        logger.warning(f"Answer count mismatch: {len(all_answers)} answers for {total_questions} questions")
        while len(all_answers) < total_questions:
            all_answers.append("Answer not available due to processing error")
        all_answers = all_answers[:total_questions]
    
    return all_answers

def query_gemini_batch(content: str, questions: List[str]) -> List[str]:
    """Send content and all questions to Gemini API with key rotation and error handling."""
    logger.info(f"Starting Gemini API batch call with {len(questions)} questions")
    logger.debug(f"Document content length: {len(content)} characters")
    logger.debug(f"Questions: {questions}")
    
    # Get current key info
    key_info = key_manager.get_current_key_info()
    logger.info(f"Using API key {key_info['key_index'] + 1}/{key_info['total_keys']} ({key_info['key_preview']}) - attempt {key_info['retry_count'] + 1}")
    
    # Format questions with numbers for better structure
    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    prompt = f"""Document Content:
{content}

Please answer the following questions based ONLY on the information provided in the document above. 
For each question, provide a direct and accurate answer. If the information is not available in the document, state "Information not available in the document."

Questions:
{questions_text}

Please respond with a JSON object in the following format:
{{
    "answers": [
        "Answer to question 1",
        "Answer to question 2",
        "Answer to question 3"
    ]
}}

Make sure your response is valid JSON and contains exactly {len(questions)} answers in the same order as the questions."""
    
    # Try with current key, rotate on failure
    max_rotation_attempts = len(key_manager.api_keys) * key_manager.max_retries_per_key
    
    for attempt in range(max_rotation_attempts):
        current_key = key_manager.get_current_key()
        url = f"{settings.GEMINI_BASE_URL}?key={current_key}"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": settings.GEMINI_TEMPERATURE,
                "maxOutputTokens": settings.GEMINI_MAX_OUTPUT_TOKENS
            }
        }
        
        logger.debug(f"Attempt {attempt + 1}/{max_rotation_attempts} - Gemini API URL: {url}")
        logger.debug(f"Payload size: {len(json.dumps(payload))} characters")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        try:
            logger.info(f"Sending request to Gemini API (attempt {attempt + 1})...")
            response = requests.post(
                url, 
                headers=headers, 
                data=json.dumps(payload), 
                timeout=settings.GEMINI_API_TIMEOUT
            )
            
            logger.info(f"Gemini API response status: {response.status_code}")
            logger.debug(f"Response headers: {dict(response.headers)}")
            
            # Check for specific error codes that indicate key issues
            if response.status_code in [401, 403]:
                logger.error(f"Authentication/Authorization error with key {current_key[:10]}... - Status: {response.status_code}")
                if key_manager.mark_key_failed(current_key):
                    logger.info("Rotated to next key due to auth error")
                    continue
                else:
                    logger.error("No more keys available for rotation")
                    break
            elif response.status_code == 429:
                logger.warning(f"Rate limit hit with key {current_key[:10]}... - attempting rotation")
                if key_manager.mark_key_failed(current_key):
                    logger.info("Rotated to next key due to rate limit")
                    continue
                else:
                    logger.error("No more keys available for rotation")
                    break
            
            # Log the raw response for debugging
            if response.status_code != 200:
                logger.error(f"Gemini API error status {response.status_code}: {response.text}")
            
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Response JSON keys: {list(result.keys())}")
            
            # If we get here, the request was successful - reset failures for this key
            key_manager.reset_key_failures(current_key)
            
            # Extract the generated text from Gemini response
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                logger.debug(f"Candidate keys: {list(candidate.keys())}")
                
                if "content" in candidate and "parts" in candidate["content"]:
                    response_text = candidate["content"]["parts"][0]["text"].strip()
                    logger.info(f"Generated response length: {len(response_text)} characters")
                    logger.debug(f"Raw response text: {response_text}")
                    
                    # Try to parse the JSON response
                    try:
                        # Clean the response - remove any markdown formatting
                        cleaned_text = response_text
                        if "```json" in response_text:
                            logger.debug("Found JSON markdown formatting, extracting...")
                            cleaned_text = response_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in response_text:
                            logger.debug("Found markdown formatting, extracting...")
                            cleaned_text = response_text.split("```")[1].split("```")[0].strip()
                        
                        logger.debug(f"Cleaned text for JSON parsing: {cleaned_text}")
                        parsed_response = json.loads(cleaned_text)
                        logger.debug(f"Parsed JSON keys: {list(parsed_response.keys())}")
                        
                        if "answers" in parsed_response and isinstance(parsed_response["answers"], list):
                            answers = parsed_response["answers"]
                            logger.info(f"Successfully extracted {len(answers)} answers from JSON response")
                            
                            # Ensure we have the right number of answers
                            if len(answers) == len(questions):
                                logger.info("Answer count matches question count")
                                return answers
                            else:
                                logger.warning(f"Answer count mismatch: {len(answers)} answers for {len(questions)} questions")
                                # If we don't have the right number, pad or truncate
                                while len(answers) < len(questions):
                                    answers.append("Answer not provided")
                                    logger.debug("Added placeholder answer")
                                return answers[:len(questions)]
                        else:
                            logger.error("Invalid JSON structure - missing 'answers' array")
                            logger.debug(f"Parsed response structure: {parsed_response}")
                            raise ValueError("Invalid JSON structure - missing 'answers' array")
                            
                    except json.JSONDecodeError as json_error:
                        # If JSON parsing fails, try to extract answers manually
                        logger.error(f"JSON parsing failed: {json_error}")
                        logger.debug(f"Raw response for fallback parsing: {response_text}")
                        
                        # Fallback: try to split the response by lines and extract answers
                        lines = response_text.split('\n')
                        answers = []
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('{') and not line.startswith('}') and not line.startswith('"answers"'):
                                # Try to find lines that look like answers
                                if any(char.isalpha() for char in line):
                                    clean_line = line.strip('",').strip()
                                    if clean_line:
                                        answers.append(clean_line)
                                        logger.debug(f"Extracted fallback answer: {clean_line}")
                        
                        # Ensure we have the right number of answers
                        while len(answers) < len(questions):
                            answers.append("Could not parse answer from response")
                            logger.debug("Added fallback error message")
                        
                        logger.warning(f"Used fallback parsing, extracted {len(answers)} answers")
                        return answers[:len(questions)]
                else:
                    logger.error(f"Missing content/parts in candidate response: {candidate}")
                    raise ValueError("Missing content in Gemini API response")
            else:
                logger.error(f"No candidates in Gemini response: {result}")
                raise ValueError("No candidates in Gemini API response")
                
        except requests.exceptions.Timeout as e:
            logger.error(f"Gemini API request timeout with key {current_key[:10]}...: {e}")
            if key_manager.mark_key_failed(current_key):
                logger.info("Rotated to next key due to timeout")
                continue
            else:
                logger.error("No more keys available after timeout")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Gemini API request timed out - all keys exhausted"
                )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Gemini API connection error with key {current_key[:10]}...: {e}")
            if key_manager.mark_key_failed(current_key):
                logger.info("Rotated to next key due to connection error")
                continue
            else:
                logger.error("No more keys available after connection error")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Unable to connect to Gemini API - all keys exhausted"
                )
        except requests.exceptions.HTTPError as e:
            logger.error(f"Gemini API HTTP error with key {current_key[:10]}...: {e}")
            # Don't rotate for client errors (4xx) unless it's auth/rate limit
            if response.status_code < 500 and response.status_code not in [401, 403, 429]:
                if response.status_code == 400:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid request to Gemini API - content may be too large"
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Gemini API returned error {response.status_code}"
                    )
            else:
                # Server errors - try to rotate
                if key_manager.mark_key_failed(current_key):
                    logger.info("Rotated to next key due to server error")
                    continue
                else:
                    logger.error("No more keys available after server error")
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Gemini API server error {response.status_code} - all keys exhausted"
                    )
        except requests.RequestException as e:
            logger.error(f"Gemini API request error with key {current_key[:10]}...: {type(e).__name__}: {e}")
            if key_manager.mark_key_failed(current_key):
                logger.info("Rotated to next key due to request error")
                continue
            else:
                logger.error("No more keys available after request error")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Error communicating with Gemini API - all keys exhausted: {str(e)}"
                )
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Gemini API with key {current_key[:10]}...: {e}")
            # JSON decode errors might be temporary, try next key
            if key_manager.mark_key_failed(current_key):
                logger.info("Rotated to next key due to JSON decode error")
                continue
            else:
                logger.error("No more keys available after JSON decode error")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Received invalid JSON response from Gemini API - all keys exhausted"
                )
        except Exception as e:
            logger.error(f"Unexpected error in Gemini API call with key {current_key[:10]}...: {type(e).__name__}: {e}")
            if key_manager.mark_key_failed(current_key):
                logger.info("Rotated to next key due to unexpected error")
                continue
            else:
                logger.error("No more keys available after unexpected error")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Unexpected error processing Gemini response - all keys exhausted: {str(e)}"
                )
    
    # If we get here, all attempts failed
    logger.error("All API keys and retry attempts exhausted")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="All Gemini API keys exhausted - service temporarily unavailable"
    )

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_pdf_questions(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Process PDF document and answer questions using Gemini API."""
    request_id = f"{hash(str(request.documents))}_{len(request.questions)}"
    logger.info(f"Processing request {request_id} with {len(request.questions)} questions")
    logger.debug(f"PDF URL: {request.documents}")
    logger.debug(f"Questions: {request.questions}")
    
    try:
        # Extract text from PDF
        logger.info(f"Request {request_id}: Starting PDF text extraction")
        pdf_content = extract_pdf_text_from_url(str(request.documents))
        
        if not pdf_content:
            logger.error(f"Request {request_id}: No text extracted from PDF")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the PDF"
            )
        
        logger.info(f"Request {request_id}: PDF text extraction successful, {len(pdf_content)} characters")
        
        # Process all questions in configurable batches
        logger.info(f"Request {request_id}: Starting batch question processing with batch size {settings.GEMINI_BATCH_SIZE}")
        answers = process_questions_in_batches(pdf_content, request.questions)
        
        logger.info(f"Request {request_id}: Successfully processed all questions, returning {len(answers)} answers")
        logger.debug(f"Request {request_id}: Answers: {answers}")
        
        return QueryResponse(answers=answers)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Request {request_id}: Unexpected error: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error processing request: {str(e)}"
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # For any other unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error processing questions: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/key-stats")
async def get_key_stats():
    """Get API key rotation statistics."""
    return {
        "key_manager_stats": key_manager.get_stats(),
        "current_key_info": key_manager.get_current_key_info()
    }

@app.post("/reset-keys")
async def reset_key_failures():
    """Reset failure counts for all API keys."""
    key_manager.reset_key_failures()
    return {
        "message": "All API key failure counts have been reset",
        "stats": key_manager.get_stats()
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "endpoints": {
            "main": "POST /hackrx/run",
            "health": "GET /health",
            "key_stats": "GET /key-stats",
            "reset_keys": "POST /reset-keys"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
