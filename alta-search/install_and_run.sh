#!/bin/bash
# Installation and run script for Kalenike Similarity Search

echo "========================================="
echo "Kalenike Similarity Search - Installation"
echo "========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Using existing virtual environment: $VIRTUAL_ENV"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "Starting Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "To stop the app, press Ctrl+C"
echo ""

# Run Streamlit app
streamlit run streamlit_app.py

