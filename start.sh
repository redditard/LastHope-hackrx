#!/bin/bash
# Startup script for PDF to Gemini Q&A API

echo "🚀 Starting PDF to Gemini Q&A API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "   Please copy .env.example to .env and add your Gemini API key"
    echo "   Example: cp .env.example .env"
    exit 1
fi

# Start the server
echo "🌟 Starting FastAPI server..."
python main.py
