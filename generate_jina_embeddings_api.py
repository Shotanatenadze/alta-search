#!/usr/bin/env python3
"""
Generate Jina embeddings v3 using Jina API for:
1. Raw data: item_name + all item_features
2. General info: item_name + general_info_features only
"""

import pandas as pd
import numpy as np
import json
import pickle
import requests
from pathlib import Path
from tqdm import tqdm
import time

# Configuration
output_dir = Path('UpdatedData')
model_dir = output_dir / 'model'
csv_file = output_dir / 'GetData_with_features.csv'

# Jina API Configuration
JINA_API_KEY = "jina_f492d2b75a6c4469bf7359bb5902a92fv1tiZ9uYwMiV-3Ux5h-1Bz_EF1Zg"
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v3"
JINA_TASK = "text-matching"

# Output file names
JINA_EMBEDDINGS_FILE = model_dir / 'jina_embeddings.npy'
JINA_MAPPING_FILE = model_dir / 'jina_item_mapping.pkl'
JINA_EMBEDDINGS_GI_FILE = model_dir / 'jina_embeddings_general_info.npy'
JINA_MAPPING_GI_FILE = model_dir / 'jina_item_mapping_general_info.pkl'

# API batch size (Jina API supports up to 2048 tokens per request, we'll batch texts)
API_BATCH_SIZE = 50  # Number of texts per API call

def extract_all_features(item_features_str):
    """
    Extract all field_name and field_value from item_features
    Returns list of tuples: [(field_name, field_value), ...]
    """
    if pd.isna(item_features_str) or not item_features_str:
        return []
    
    try:
        features_data = json.loads(item_features_str)
        features = features_data.get('features', [])
        
        all_features = []
        for feature in features:
            field_name = feature.get('field_name', '')
            field_value = feature.get('field_value', '')
            if field_name and field_value:
                all_features.append((field_name, field_value))
        
        return all_features
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error parsing item_features: {e}")
        return []

def extract_general_info_features(item_features_str):
    """
    Extract field_name and field_value from item_features where group_name == "ზოგადი ინფორმაცია"
    Returns list of tuples: [(field_name, field_value), ...]
    """
    if pd.isna(item_features_str) or not item_features_str:
        return []
    
    try:
        features_data = json.loads(item_features_str)
        features = features_data.get('features', [])
        
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

def prepare_text_for_embedding(item_name, features):
    """
    Prepare text string from item_name and features for Jina embedding
    Format: "item_name. field_name: field_value. field_name: field_value. ..."
    """
    text_parts = []
    
    # Add item_name
    if pd.notna(item_name) and str(item_name).strip():
        text_parts.append(str(item_name).strip())
    
    # Add features
    for field_name, field_value in features:
        if field_name and field_value:
            text_parts.append(f"{field_name}: {field_value}")
    
    # Join with periods and spaces
    text = ". ".join(text_parts)
    if text and not text.endswith('.'):
        text += "."
    
    return text

def call_jina_api(texts, max_retries=3):
    """
    Call Jina API to get embeddings for a batch of texts
    Returns list of embeddings (1024 dimensions each)
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    
    data = {
        "model": JINA_MODEL,
        "task": JINA_TASK,
        "input": texts
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                JINA_API_URL, 
                headers=headers, 
                data=json.dumps(data),
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            # Extract embeddings from response
            embeddings = [item['embedding'] for item in result['data']]
            return embeddings
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"API call failed after {max_retries} attempts: {e}")
                raise
    
    return None

def create_jina_embeddings(df, use_general_info_only=False):
    """
    Create Jina embeddings for all items using API
    use_general_info_only: If True, use only general_info_features. If False, use all features.
    """
    print(f"Preparing texts and creating embeddings via Jina API...")
    texts = []
    item_ids = []
    
    # Progress bar for text preparation
    pbar_prep = tqdm(total=len(df), desc="Preparing texts", unit="items")
    
    for idx, row in df.iterrows():
        item_id = row['item']
        item_name = row['item_name']
        
        if use_general_info_only:
            # Use general_info_features column if available, otherwise extract
            if 'general_info_text' in row and pd.notna(row['general_info_text']) and row['general_info_text'].strip():
                # Use pre-formatted general_info_text
                general_info_text = row['general_info_text']
                if general_info_text:
                    # Parse the formatted text back to features
                    features = []
                    parts = general_info_text.split(' | ')
                    for part in parts:
                        if ': ' in part:
                            field_name, field_value = part.split(': ', 1)
                            features.append((field_name.strip(), field_value.strip()))
                    text = prepare_text_for_embedding(item_name, features)
                else:
                    text = prepare_text_for_embedding(item_name, [])
            else:
                # Extract general info features
                features = extract_general_info_features(row['item_features'])
                text = prepare_text_for_embedding(item_name, features)
        else:
            # Use all features
            features = extract_all_features(row['item_features'])
            text = prepare_text_for_embedding(item_name, features)
        
        # Only add if we have at least item_name
        if text and text.strip():
            texts.append(text)
            item_ids.append(item_id)
        else:
            print(f"Warning: No text for item {item_id}, skipping")
        
        pbar_prep.update(1)
    
    pbar_prep.close()
    print(f"Prepared {len(texts)} texts for embedding")
    
    # Create embeddings via API in batches
    embeddings = []
    item_mapping = {}
    
    print(f"Generating embeddings via Jina API in batches of {API_BATCH_SIZE}...")
    num_batches = (len(texts) + API_BATCH_SIZE - 1) // API_BATCH_SIZE
    
    # Create progress bar
    pbar = tqdm(total=len(texts), desc="Generating embeddings", unit="items")
    
    for i in range(0, len(texts), API_BATCH_SIZE):
        batch_texts = texts[i:i+API_BATCH_SIZE]
        batch_item_ids = item_ids[i:i+API_BATCH_SIZE]
        
        # Call Jina API
        try:
            batch_embeddings = call_jina_api(batch_texts)
            
            # Store embeddings
            for j, (item_id, embedding) in enumerate(zip(batch_item_ids, batch_embeddings)):
                embeddings.append(embedding)
                item_mapping[item_id] = len(embeddings) - 1
            
            # Update progress bar
            pbar.update(len(batch_texts))
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\nError processing batch {i//API_BATCH_SIZE + 1}/{num_batches}: {e}")
            # Skip this batch but continue
            pbar.update(len(batch_texts))
            continue
    
    pbar.close()
    
    embeddings = np.array(embeddings)
    print(f"Created embeddings shape: {embeddings.shape}")
    print(f"Mapped {len(item_mapping)} items")
    
    return embeddings, item_mapping

def main():
    print("Loading data...")
    df = pd.read_csv(csv_file, encoding='utf-8')
    print(f"Loaded {len(df)} rows")
    
    # Filter to only include products with non-empty product_url (same as streamlit app)
    print("Filtering to items with product_url...")
    df = df[df['product_url'].notna() & (df['product_url'].astype(str).str.strip() != '')].copy()
    print(f"Filtered to {len(df)} items with product_url")
    
    # Ensure model directory exists
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create raw data embeddings (all features)
    print("\n" + "="*50)
    print("Creating Jina embeddings for RAW DATA (all features)...")
    print("="*50)
    jina_embeddings, jina_mapping = create_jina_embeddings(df, use_general_info_only=False)
    
    # Save raw data embeddings
    print("\nSaving raw data embeddings...")
    np.save(JINA_EMBEDDINGS_FILE, jina_embeddings)
    with open(JINA_MAPPING_FILE, 'wb') as f:
        pickle.dump(jina_mapping, f)
    print(f"Saved Jina embeddings: {JINA_EMBEDDINGS_FILE}")
    print(f"Saved Jina mapping: {JINA_MAPPING_FILE}")
    print(f"  Embeddings shape: {jina_embeddings.shape}")
    print(f"  Mapped items: {len(jina_mapping)}")
    
    # Create general info embeddings
    print("\n" + "="*50)
    print("Creating Jina embeddings for GENERAL INFO (filtered features)...")
    print("="*50)
    jina_embeddings_gi, jina_mapping_gi = create_jina_embeddings(df, use_general_info_only=True)
    
    # Save general info embeddings
    print("\nSaving general info embeddings...")
    np.save(JINA_EMBEDDINGS_GI_FILE, jina_embeddings_gi)
    with open(JINA_MAPPING_GI_FILE, 'wb') as f:
        pickle.dump(jina_mapping_gi, f)
    print(f"Saved Jina general info embeddings: {JINA_EMBEDDINGS_GI_FILE}")
    print(f"Saved Jina general info mapping: {JINA_MAPPING_GI_FILE}")
    print(f"  Embeddings shape: {jina_embeddings_gi.shape}")
    print(f"  Mapped items: {len(jina_mapping_gi)}")
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)

if __name__ == '__main__':
    main()

