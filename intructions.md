# PDF to Gemini Q\&A Flow (No Chunking Approach)

## Goal

Implement a **simple pipeline** that:

1. Extracts all text from a PDF.
2. Sends the extracted content directly to Gemini.
3. Asks questions about the content.

---

## Steps

### 1. **Extract Text from PDF**

Use `pypdf` or `pdfplumber` in Python.

```python
import pdfplumber

def extract_pdf_text(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text
```

### 2. **Send to Gemini API**

Assuming you have an API key stored in `settings.GEMINI_API_KEY`.

```python
import requests
import json

def query_gemini(content, question):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={settings.GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": f"Document Content:\n{content}\n\nQuestion: {question}"}]}]
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()
```

### 3. **Main Execution**

```python
if __name__ == "__main__":
    pdf_path = "policy.pdf"
    question = "What is the waiting period mentioned in this document?"
    
    content = extract_pdf_text(pdf_path)
    answer = query_gemini(content, question)
    print(answer)
```

---

## Notes & Warnings

* **Token Limit Risk**: If the PDF exceeds Gemini's context window (\~8K tokens for standard Gemini Pro), part of the text will be truncated.
* **Accuracy Impact**: Large documents without chunking may cause the model to miss relevant details.
* **API Costs**: Sending large text in one request will increase token usage.
* **Recommendation**: For production, implement chunking & retrieval.

---

## Optional Improvements

* Add OCR for scanned PDFs using `pytesseract`.
* Clean and normalize whitespace & formatting before sending.
* Implement a size check and warn if PDF is likely too large for the model.
