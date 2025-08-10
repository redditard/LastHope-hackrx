#!/usr/bin/env python3
"""
Test script for the PDF to Gemini Q&A API
"""

import requests
import json
import os

# API Configuration
BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "09988a27dc0bf3ef755e893e2e6650693e4009189215f7824b023cc07db59b1b"

# Test data from API spec
TEST_PDF_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

TEST_QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?"
]

def test_health_check():
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_main_endpoint():
    """Test the main Q&A endpoint."""
    print("\nTesting main Q&A endpoint...")
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": TEST_PDF_URL,
        "questions": TEST_QUESTIONS
    }
    
    try:
        print("Sending request...")
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=headers,
            data=json.dumps(payload),
            timeout=120  # Increased timeout for PDF processing
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print("\nAnswers:")
            for i, answer in enumerate(result["answers"], 1):
                print(f"\n{i}. Question: {TEST_QUESTIONS[i-1]}")
                print(f"   Answer: {answer}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_invalid_token():
    """Test with invalid bearer token."""
    print("\nTesting invalid token...")
    
    headers = {
        "Authorization": "Bearer invalid_token",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": TEST_PDF_URL,
        "questions": ["Test question"]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=headers,
            data=json.dumps(payload)
        )
        
        if response.status_code == 401:
            print("✅ Correctly rejected invalid token")
        else:
            print(f"❌ Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def main():
    """Run all tests."""
    print("🧪 Starting API Tests")
    print("=" * 50)
    
    # Check if server is running
    if not test_health_check():
        print("\n❌ Server is not running. Please start the server first:")
        print("   python main.py")
        return
    
    # Test main functionality
    test_main_endpoint()
    
    # Test security
    test_invalid_token()
    
    print("\n" + "=" * 50)
    print("🏁 Tests completed!")

if __name__ == "__main__":
    main()
