"""
Configuration settings for the PDF to Gemini Q&A API
"""
import os
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # Authentication
    VALID_BEARER_TOKEN: str = "09988a27dc0bf3ef755e893e2e6650693e4009189215f7824b023cc07db59b1b"
    
    # Gemini API
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_API_KEYS: List[str] = []
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
    
    # Key rotation settings
    GEMINI_KEY_ROTATION_ENABLED: bool = os.getenv("GEMINI_KEY_ROTATION_ENABLED", "true").lower() == "true"
    GEMINI_MAX_RETRIES_PER_KEY: int = int(os.getenv("GEMINI_MAX_RETRIES_PER_KEY", "2"))
    
    # API Settings
    API_TITLE: str = "PDF to Gemini Q&A API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Extract text from PDFs and answer questions using Google Gemini AI"
    
    # Request timeouts (seconds)
    PDF_DOWNLOAD_TIMEOUT: int = 30
    GEMINI_API_TIMEOUT: int = 30
    
    # Gemini generation config
    GEMINI_TEMPERATURE: float = 0.1
    GEMINI_MAX_OUTPUT_TOKENS: int = 1000
    GEMINI_BATCH_SIZE: int = int(os.getenv("GEMINI_BATCH_SIZE", "10"))  # Questions per batch
    
    # Logging configuration
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    LOG_FILE: str = "api.log"
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Initialize API keys list for rotation
        self.GEMINI_API_KEYS = [self.GEMINI_API_KEY]  # Start with primary key
        
        # Add additional keys if provided
        additional_keys = os.getenv("GEMINI_API_KEYS", "")
        if additional_keys:
            extra_keys = [key.strip() for key in additional_keys.split(",") if key.strip()]
            self.GEMINI_API_KEYS.extend(extra_keys)
        
        print(f"Initialized {len(self.GEMINI_API_KEYS)} API key(s) for rotation")

# Global settings instance
settings = Settings()
