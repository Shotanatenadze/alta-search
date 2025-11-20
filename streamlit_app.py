#!/usr/bin/env python3
"""
Streamlit application for searching similar items using Word2Vec and FastText embeddings
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
output_dir = Path('UpdatedData')
model_dir = output_dir / 'model'

# Load data and embeddings
@st.cache_data
def load_data():
    """Load items data"""
    df = pd.read_csv(output_dir / 'GetData_with_features.csv', encoding='utf-8')
    return df

@st.cache_resource
def load_word2vec_embeddings():
    """Load Word2Vec embeddings and mapping"""
    embeddings = np.load(model_dir / 'word2vec_embeddings.npy')
    with open(model_dir / 'word2vec_item_mapping.pkl', 'rb') as f:
        item_mapping = pickle.load(f)
    return embeddings, item_mapping

@st.cache_resource
def load_fasttext_embeddings():
    """Load FastText embeddings and mapping"""
    embeddings = np.load(model_dir / 'fasttext_embeddings.npy')
    with open(model_dir / 'fasttext_item_mapping.pkl', 'rb') as f:
        item_mapping = pickle.load(f)
    return embeddings, item_mapping

def find_similar_items(query_item_id, embeddings, item_mapping, df, top_k=15):
    """Find similar items within the same m2 category"""
    # Get query item's m2 category
    query_item = df[df['item'] == query_item_id]
    if query_item.empty:
        return []
    
    query_m2 = query_item.iloc[0]['m2']
    
    # Get query embedding from pre-computed embeddings
    if query_item_id not in item_mapping:
        return []
    
    query_embedding_idx = item_mapping[query_item_id]
    query_embedding = embeddings[query_embedding_idx].reshape(1, -1)
    
    # Filter items by same m2 category
    same_category_df = df[df['m2'] == query_m2].copy()
    
    if same_category_df.empty:
        return []
    
    # Get embeddings for items in same category
    same_category_items = same_category_df['item'].values
    item_indices = []
    category_embeddings = []
    
    for item_id in same_category_items:
        if item_id in item_mapping:
            idx = item_mapping[item_id]
            item_indices.append(item_id)
            category_embeddings.append(embeddings[idx])
    
    if not category_embeddings:
        return []
    
    category_embeddings = np.array(category_embeddings)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, category_embeddings)[0]
    
    # Get top k (excluding the query item itself)
    top_indices = np.argsort(similarities)[::-1]
    results = []
    
    # Columns to include
    columns_to_include = ['item', 'item_model', 'brand', 'category', 'item_name', 'product_url', 'bcat1', 'bcat2', 'm1', 'm2', 'm3']
    
    for idx in top_indices:
        item_id = item_indices[idx]
        if item_id != query_item_id:  # Exclude query item
            similarity = similarities[idx]
            item_data = same_category_df[same_category_df['item'] == item_id].iloc[0]
            
            # Create result dict with all requested columns
            result = {'similarity': similarity}
            for col in columns_to_include:
                if col in item_data:
                    value = item_data[col]
                    # Handle NaN values
                    if pd.isna(value):
                        result[col] = ""
                    else:
                        result[col] = str(value)
                else:
                    result[col] = ""
            
            results.append(result)
            if len(results) >= top_k:
                break
    
    return results

# Streamlit App
st.set_page_config(page_title="Item Similarity Search", layout="wide")

st.title("🔍 Item Similarity Search")
st.markdown("Search for similar items using Word2Vec and FastText embeddings (optimized for Georgian language)")

# Load data
df = load_data()

# Search interface
col1, col2 = st.columns([2, 1])

with col1:
    search_input = st.text_input("Search by Item ID or Name", placeholder="Enter item ID (e.g., 11960) or item name")

with col2:
    top_k = st.number_input("Number of results", min_value=1, max_value=50, value=15)

if search_input:
    # Try to find item by ID or name
    query_item = None
    
    # Try as item ID first
    try:
        item_id = int(search_input)
        query_item = df[df['item'] == item_id]
    except ValueError:
        # Search by name
        query_item = df[df['item_name'].str.contains(search_input, case=False, na=False)]
    
    if query_item.empty:
        st.error(f"No item found matching: {search_input}")
    else:
        # Use first match
        selected_item = query_item.iloc[0]
        query_item_id = selected_item['item']
        
        # Display query item with all columns
        st.markdown("---")
        st.subheader(f"📦 Query Item: {selected_item['item_name']}")
        
        # Create a nice display of all columns
        columns_to_show = ['item', 'item_model', 'brand', 'category', 'item_name', 'product_url', 'bcat1', 'bcat2', 'm1', 'm2', 'm3']
        
        # Display in a table format
        query_data = {}
        for col in columns_to_show:
            if col in selected_item:
                value = selected_item[col]
                query_data[col] = "" if pd.isna(value) else str(value)
            else:
                query_data[col] = ""
        
        # Display in columns for better layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Basic Info**")
            st.text(f"Item ID: {query_data['item']}")
            st.text(f"Item Name: {query_data['item_name']}")
            st.text(f"Item Model: {query_data['item_model']}")
            st.text(f"Brand: {query_data['brand']}")
        
        with col2:
            st.markdown("**Categories**")
            st.text(f"Category: {query_data['category']}")
            st.text(f"BCat1: {query_data['bcat1']}")
            st.text(f"BCat2: {query_data['bcat2']}")
            st.text(f"M1: {query_data['m1']}")
        
        with col3:
            st.markdown("**Hierarchy**")
            st.text(f"M2: {query_data['m2']}")
            st.text(f"M3: {query_data['m3']}")
            if query_data['product_url']:
                st.markdown(f"**Product URL:**")
                st.markdown(f"[Link]({query_data['product_url']})")
        
        # Load embeddings
        with st.spinner("Loading embeddings..."):
            w2v_embeddings, w2v_mapping = load_word2vec_embeddings()
            fasttext_embeddings, fasttext_mapping = load_fasttext_embeddings()
        
        # Find similar items
        st.markdown("---")
        st.subheader("🔎 Similar Items")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Word2Vec Results")
            with st.spinner("Finding similar items with Word2Vec..."):
                w2v_results = find_similar_items(
                    query_item_id, w2v_embeddings, w2v_mapping, 
                    df, top_k=top_k
                )
            
            if w2v_results:
                for i, result in enumerate(w2v_results, 1):
                    with st.expander(f"#{i} {result['item_name']} (Similarity: {result['similarity']:.4f})", expanded=False):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown("**Basic Info**")
                            st.text(f"Item ID: {result['item']}")
                            st.text(f"Item Name: {result['item_name']}")
                            st.text(f"Item Model: {result['item_model']}")
                            st.text(f"Brand: {result['brand']}")
                        with col_b:
                            st.markdown("**Categories**")
                            st.text(f"Category: {result['category']}")
                            st.text(f"BCat1: {result['bcat1']}")
                            st.text(f"BCat2: {result['bcat2']}")
                            st.text(f"M1: {result['m1']}")
                        with col_c:
                            st.markdown("**Hierarchy**")
                            st.text(f"M2: {result['m2']}")
                            st.text(f"M3: {result['m3']}")
                            if result['product_url']:
                                st.markdown(f"**Product URL:**")
                                st.markdown(f"[Link]({result['product_url']})")
            else:
                st.info("No similar items found in the same category")
        
        with col2:
            st.markdown("### FastText Results")
            with st.spinner("Finding similar items with FastText..."):
                fasttext_results = find_similar_items(
                    query_item_id, fasttext_embeddings, fasttext_mapping, 
                    df, top_k=top_k
                )
            
            if fasttext_results:
                for i, result in enumerate(fasttext_results, 1):
                    with st.expander(f"#{i} {result['item_name']} (Similarity: {result['similarity']:.4f})", expanded=False):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown("**Basic Info**")
                            st.text(f"Item ID: {result['item']}")
                            st.text(f"Item Name: {result['item_name']}")
                            st.text(f"Item Model: {result['item_model']}")
                            st.text(f"Brand: {result['brand']}")
                        with col_b:
                            st.markdown("**Categories**")
                            st.text(f"Category: {result['category']}")
                            st.text(f"BCat1: {result['bcat1']}")
                            st.text(f"BCat2: {result['bcat2']}")
                            st.text(f"M1: {result['m1']}")
                        with col_c:
                            st.markdown("**Hierarchy**")
                            st.text(f"M2: {result['m2']}")
                            st.text(f"M3: {result['m3']}")
                            if result['product_url']:
                                st.markdown(f"**Product URL:**")
                                st.markdown(f"[Link]({result['product_url']})")
            else:
                st.info("No similar items found in the same category")

else:
    st.info("👆 Enter an item ID or name to search for similar items")
    st.markdown("### Sample Items:")
    sample_items = df.head(10)[['item', 'item_name', 'brand', 'category']]
    st.dataframe(sample_items, width='stretch')

