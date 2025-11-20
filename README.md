# Kalenike Item Similarity Search

A ready-to-run Streamlit application for finding similar items using Word2Vec and FastText embeddings.

## Quick Start

### Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

The app will open in your browser at `http://localhost:8501`

### Using the Installation Script

Alternatively, use the provided script:

```bash
chmod +x install_and_run.sh
./install_and_run.sh
```

## Features

- 🔍 Search items by ID or name
- 📊 Find top 15 similar items using Word2Vec and FastText models
- 🎯 Filter results by same m2 category
- 📋 Display comprehensive item information

## Requirements

- Python 3.8 or higher
- All dependencies are listed in `requirements.txt`

## Troubleshooting

### Port already in use
Use a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Missing dependencies
Make sure you installed requirements:
```bash
pip install -r requirements.txt
```

## Files Included

- `streamlit_app.py` - Main application
- `UpdatedData/GetData_with_features.csv` - Item data
- `UpdatedData/model/` - Pre-trained models and embeddings
- `requirements.txt` - Python dependencies

## Support

For issues or questions, contact the development team.

