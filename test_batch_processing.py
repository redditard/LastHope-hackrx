#!/usr/bin/env python3
"""
Test script to demonstrate batch processing functionality
"""

import requests
import json
import time

# API Configuration
BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "09988a27dc0bf3ef755e893e2e6650693e4009189215f7824b023cc07db59b1b"

# Test data
TEST_PDF_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

# Test with many questions to demonstrate batching
MANY_QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?",
    "What is the maximum age for entry into this policy?",
    "What is the minimum age for coverage under this policy?",
    "Are there any geographical restrictions for coverage?",
    "What documentation is required for claim settlement?",
    "What is the process for cashless treatment?"
]

def test_batch_processing():
    """Test the API with many questions to see batch processing in action."""
    print("🧪 Testing Batch Processing Functionality")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": TEST_PDF_URL,
        "questions": MANY_QUESTIONS
    }
    
    print(f"Sending {len(MANY_QUESTIONS)} questions to test batch processing...")
    print("Check the server logs to see how questions are batched!")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=headers,
            data=json.dumps(payload),
            timeout=300  # Longer timeout for multiple batches
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answers = result["answers"]
            print(f"✅ Success! Received {len(answers)} answers")
            
            print("\n📝 Sample Answers (first 3):")
            for i, (question, answer) in enumerate(zip(MANY_QUESTIONS[:3], answers[:3]), 1):
                print(f"\n{i}. Q: {question}")
                print(f"   A: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            
            if len(answers) > 3:
                print(f"\n... and {len(answers) - 3} more answers")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_small_batch():
    """Test with a small number of questions (single batch)."""
    print("\n" + "=" * 50)
    print("🧪 Testing Small Batch (Single API Call)")
    print("=" * 50)
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Content-Type": "application/json"
    }
    
    small_questions = MANY_QUESTIONS[:3]  # Just 3 questions
    
    payload = {
        "documents": TEST_PDF_URL,
        "questions": small_questions
    }
    
    print(f"Sending {len(small_questions)} questions (should be processed in single batch)...")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=headers,
            data=json.dumps(payload),
            timeout=120
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"📊 Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Received {len(result['answers'])} answers")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Batch Processing Tests")
    print("💡 Check the server logs to see batch processing in action!")
    print()
    
    # Test small batch first
    test_small_batch()
    
    # Then test large batch
    test_batch_processing()
    
    print("\n" + "=" * 50)
    print("🏁 Tests completed!")
    print("💡 Tip: You can adjust GEMINI_BATCH_SIZE in your .env file")
    print("   Current batch size can be seen in the server startup logs")
