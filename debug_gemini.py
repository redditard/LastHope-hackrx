#!/usr/bin/env python3
"""
Test script to debug Gemini API issues
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(f"API Key loaded: {GEMINI_API_KEY[:10]}...{GEMINI_API_KEY[-10:] if GEMINI_API_KEY else 'None'}")

def test_gemini_api():
    """Test a simple Gemini API call."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Hello, please respond with 'API is working correctly'"
                    }
                ]
            }
        ]
    }
    
    try:
        print("Making test API call to Gemini...")
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Gemini API call successful!")
            print(f"Full Response: {json.dumps(result, indent=2)}")
            
            # Extract the response
            if "candidates" in result and len(result["candidates"]) > 0:
                if "content" in result["candidates"][0] and "parts" in result["candidates"][0]["content"]:
                    answer = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                    print(f"Generated Text: {answer}")
                else:
                    print("❌ Unexpected response structure")
            else:
                print("❌ No candidates in response")
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {type(e).__name__}: {str(e)}")

def test_pdf_download():
    """Test PDF download and text extraction."""
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        print("Testing PDF download...")
        response = requests.get(pdf_url, timeout=30)
        print(f"PDF Download Status: {response.status_code}")
        print(f"Content Length: {len(response.content)} bytes")
        
        if response.status_code == 200:
            print("✅ PDF downloaded successfully")
            
            # Test text extraction
            import pdfplumber
            from io import BytesIO
            
            print("Testing text extraction...")
            text = ""
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                print(f"PDF has {len(pdf.pages)} pages")
                for i, page in enumerate(pdf.pages[:2]):  # Only first 2 pages for testing
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        print(f"Page {i+1}: {len(page_text)} characters extracted")
            
            print(f"Total text extracted: {len(text)} characters")
            print(f"First 200 characters: {text[:200]}...")
            print("✅ Text extraction successful")
            
        else:
            print(f"❌ PDF download failed: {response.text}")
            
    except Exception as e:
        print(f"❌ PDF processing failed: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    print("🔍 Debugging Gemini API Integration")
    print("=" * 50)
    
    test_gemini_api()
    print("\n" + "-" * 50 + "\n")
    test_pdf_download()
