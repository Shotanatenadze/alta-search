#!/usr/bin/env python3
"""
Generate Word2Vec and FastText embeddings using only:
- item_name column
- From item_features: field_name and field_value where group_name == "ზოგადი ინფორმაცია"
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from gensim.models import Word2Vec, FastText
import re

# Configuration
output_dir = Path('UpdatedData')
model_dir = output_dir / 'model'
csv_file = output_dir / 'GetData_with_features.csv'

# Output file names (with suffix to distinguish from original)
WORD2VEC_EMBEDDINGS_FILE = model_dir / 'word2vec_embeddings_general_info.npy'
WORD2VEC_MAPPING_FILE = model_dir / 'word2vec_item_mapping_general_info.pkl'
FASTTEXT_EMBEDDINGS_FILE = model_dir / 'fasttext_embeddings_general_info.npy'
FASTTEXT_MAPPING_FILE = model_dir / 'fasttext_item_mapping_general_info.pkl'

def extract_general_info_features(item_features_str):
    """
    Extract field_name and field_value from item_features where group_name == "ზოგადი ინფორმაცია"
    Returns list of tuples: [(field_name, field_value), ...]
    """
    if pd.isna(item_features_str) or not item_features_str:
        return []
    
    try:
        # Parse JSON string
        features_data = json.loads(item_features_str)
        features = features_data.get('features', [])
        
        # Filter by group_name
        general_info_features = []
        for feature in features:
            if feature.get('group_name') == 'ზოგადი ინფორმაცია':
                field_name = feature.get('field_name', '')
                field_value = feature.get('field_value', '')
                if field_name and field_value:
                    general_info_features.append((field_name, field_value))
        
        return general_info_features
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error parsing item_features: {e}")
        return []

def format_general_info_text(general_info_features):
    """
    Format general info features as a text string for the CSV column
    Format: "field_name: field_value | field_name: field_value | ..."
    """
    if not general_info_features:
        return ""
    
    parts = []
    for field_name, field_value in general_info_features:
        parts.append(f"{field_name}: {field_value}")
    
    return " | ".join(parts)

def tokenize_text(text):
    """
    Simple tokenization - split by whitespace and punctuation
    """
    if pd.isna(text) or not text:
        return []
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Split by whitespace and punctuation, keep only alphanumeric tokens
    tokens = re.findall(r'\b\w+\b', text)
    
    return tokens

def prepare_text_data(df):
    """
    Prepare text data for training: combine item_name with filtered features
    """
    texts = []
    item_ids = []
    
    for idx, row in df.iterrows():
        item_id = row['item']
        item_name = row['item_name']
        
        # Extract general info features
        general_info = extract_general_info_features(row['item_features'])
        
        # Combine item_name with field_name and field_value
        text_parts = []
        
        # Add item_name tokens
        item_name_tokens = tokenize_text(item_name)
        text_parts.extend(item_name_tokens)
        
        # Add field_name and field_value from general info
        for field_name, field_value in general_info:
            field_name_tokens = tokenize_text(field_name)
            field_value_tokens = tokenize_text(field_value)
            text_parts.extend(field_name_tokens)
            text_parts.extend(field_value_tokens)
        
        # Only add if we have at least some tokens
        if text_parts:
            texts.append(text_parts)
            item_ids.append(item_id)
        else:
            print(f"Warning: No tokens for item {item_id}, skipping")
    
    return texts, item_ids

def train_word2vec(texts, vector_size=100, window=5, min_count=1, workers=4):
    """
    Train Word2Vec model
    """
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=0  # CBOW
    )
    return model

def train_fasttext(texts, vector_size=100, window=5, min_count=1, workers=4):
    """
    Train FastText model
    """
    print("Training FastText model...")
    model = FastText(
        sentences=texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=0  # CBOW
    )
    return model

def get_document_embedding(text_tokens, model):
    """
    Get document embedding by averaging word embeddings
    """
    word_vectors = []
    for token in text_tokens:
        if token in model.wv:
            word_vectors.append(model.wv[token])
    
    if not word_vectors:
        # Return zero vector if no words found
        return np.zeros(model.vector_size)
    
    return np.mean(word_vectors, axis=0)

def create_embeddings(model, texts, item_ids):
    """
    Create embeddings for all items
    """
    print(f"Creating embeddings for {len(texts)} items...")
    embeddings = []
    item_mapping = {}
    
    for idx, (text_tokens, item_id) in enumerate(zip(texts, item_ids)):
        embedding = get_document_embedding(text_tokens, model)
        embeddings.append(embedding)
        item_mapping[item_id] = idx
    
    embeddings = np.array(embeddings)
    return embeddings, item_mapping

def main():
    print("Loading data...")
    df = pd.read_csv(csv_file, encoding='utf-8')
    print(f"Loaded {len(df)} rows")
    
    # Extract general info features and add as new column
    print("\n" + "="*50)
    print("Extracting general info features...")
    print("="*50)
    
    general_info_json_list = []
    general_info_text_list = []
    
    for idx, row in df.iterrows():
        general_info = extract_general_info_features(row['item_features'])
        # Convert to list of dicts for JSON storage
        general_info_dicts = [{'field_name': fn, 'field_value': fv} for fn, fv in general_info]
        general_info_json = json.dumps(general_info_dicts, ensure_ascii=False) if general_info_dicts else ""
        general_info_json_list.append(general_info_json)
        
        general_info_text = format_general_info_text(general_info)
        general_info_text_list.append(general_info_text)
    
    # Add new columns to dataframe
    df['general_info_features'] = general_info_json_list
    df['general_info_text'] = general_info_text_list
    
    # Save updated CSV
    print("Saving updated CSV with general_info_features and general_info_text columns...")
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"Saved updated CSV: {csv_file}")
    print(f"  Added 'general_info_features' column (JSON format)")
    print(f"  Added 'general_info_text' column with {sum(1 for x in general_info_text_list if x)} non-empty entries")
    
    # Prepare text data
    print("\n" + "="*50)
    print("Preparing text data for embeddings...")
    print("="*50)
    texts, item_ids = prepare_text_data(df)
    print(f"Prepared {len(texts)} text samples")
    
    if not texts:
        print("Error: No text data prepared. Exiting.")
        return
    
    # Train models
    print("\n" + "="*50)
    print("Training models...")
    print("="*50)
    
    # Word2Vec
    w2v_model = train_word2vec(texts, vector_size=100, window=5, min_count=1, workers=4)
    print(f"Word2Vec vocabulary size: {len(w2v_model.wv)}")
    
    # FastText
    fasttext_model = train_fasttext(texts, vector_size=100, window=5, min_count=1, workers=4)
    print(f"FastText vocabulary size: {len(fasttext_model.wv)}")
    
    # Create embeddings
    print("\n" + "="*50)
    print("Creating embeddings...")
    print("="*50)
    
    w2v_embeddings, w2v_mapping = create_embeddings(w2v_model, texts, item_ids)
    fasttext_embeddings, fasttext_mapping = create_embeddings(fasttext_model, texts, item_ids)
    
    # Save embeddings
    print("\n" + "="*50)
    print("Saving embeddings...")
    print("="*50)
    
    # Ensure model directory exists
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Word2Vec
    np.save(WORD2VEC_EMBEDDINGS_FILE, w2v_embeddings)
    with open(WORD2VEC_MAPPING_FILE, 'wb') as f:
        pickle.dump(w2v_mapping, f)
    print(f"Saved Word2Vec embeddings: {WORD2VEC_EMBEDDINGS_FILE}")
    print(f"Saved Word2Vec mapping: {WORD2VEC_MAPPING_FILE}")
    print(f"  Embeddings shape: {w2v_embeddings.shape}")
    print(f"  Mapped items: {len(w2v_mapping)}")
    
    # Save FastText
    np.save(FASTTEXT_EMBEDDINGS_FILE, fasttext_embeddings)
    with open(FASTTEXT_MAPPING_FILE, 'wb') as f:
        pickle.dump(fasttext_mapping, f)
    print(f"Saved FastText embeddings: {FASTTEXT_EMBEDDINGS_FILE}")
    print(f"Saved FastText mapping: {FASTTEXT_MAPPING_FILE}")
    print(f"  Embeddings shape: {fasttext_embeddings.shape}")
    print(f"  Mapped items: {len(fasttext_mapping)}")
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)

if __name__ == '__main__':
    main()

