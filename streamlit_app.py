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
import plotly.graph_objects as go

# Import gensim to ensure it's available for pickle deserialization
# (pickle files may contain gensim object references)
try:
    import gensim
    from gensim.models import Word2Vec, FastText
except ImportError:
    pass

# Configuration
output_dir = Path('UpdatedData')
model_dir = output_dir / 'model'

# Load data and embeddings
@st.cache_data
def load_data():
    """Load items data - only products with non-empty product_url"""
    df = pd.read_csv(output_dir / 'GetData_with_features.csv', encoding='utf-8')
    # Filter to only include products with non-empty product_url
    df = df[df['product_url'].notna() & (df['product_url'].astype(str).str.strip() != '')].copy()
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

@st.cache_resource
def load_word2vec_embeddings_general_info():
    """Load Word2Vec general info embeddings and mapping"""
    embeddings = np.load(model_dir / 'word2vec_embeddings_general_info.npy')
    with open(model_dir / 'word2vec_item_mapping_general_info.pkl', 'rb') as f:
        item_mapping = pickle.load(f)
    return embeddings, item_mapping

@st.cache_resource
def load_fasttext_embeddings_general_info():
    """Load FastText general info embeddings and mapping"""
    embeddings = np.load(model_dir / 'fasttext_embeddings_general_info.npy')
    with open(model_dir / 'fasttext_item_mapping_general_info.pkl', 'rb') as f:
        item_mapping = pickle.load(f)
    return embeddings, item_mapping

@st.cache_resource
def load_jina_embeddings():
    """Load Jina embeddings and mapping"""
    embeddings = np.load(model_dir / 'jina_embeddings.npy')
    with open(model_dir / 'jina_item_mapping.pkl', 'rb') as f:
        item_mapping = pickle.load(f)
    return embeddings, item_mapping

@st.cache_resource
def load_jina_embeddings_general_info():
    """Load Jina general info embeddings and mapping"""
    embeddings = np.load(model_dir / 'jina_embeddings_general_info.npy')
    with open(model_dir / 'jina_item_mapping_general_info.pkl', 'rb') as f:
        item_mapping = pickle.load(f)
    return embeddings, item_mapping

def find_similar_items(query_item_id, embeddings, item_mapping, df, top_k=15, min_similarity=0.0, max_similarity=1.0):
    """Find similar items within the same m3 category (if defined) or same m2 category with no m3 (if m3 not defined), sorted by price closeness"""
    # Get query item's m2, m3 category and price
    query_item = df[df['item'] == query_item_id]
    if query_item.empty:
        return []
    
    query_m2 = query_item.iloc[0]['m2']
    query_m3 = query_item.iloc[0].get('m3', None)
    query_price = parse_price(query_item.iloc[0].get('price', None))
    
    # Check if m3 is defined (not null/NaN and not empty string)
    query_m3_defined = query_m3 is not None and not pd.isna(query_m3) and str(query_m3).strip() != ''
    
    # Get query embedding from pre-computed embeddings
    if query_item_id not in item_mapping:
        return []
    
    query_embedding_idx = item_mapping[query_item_id]
    query_embedding = embeddings[query_embedding_idx].reshape(1, -1)
    
    # Filter items based on m3 logic:
    # - If query has m3: filter by same m3
    # - If query has no m3: filter by same m2 AND m3 is null/empty
    if query_m3_defined:
        # Filter by same m3 category
        same_category_df = df[(df['m3'] == query_m3) & (df['m3'].notna()) & (df['m3'].astype(str).str.strip() != '')].copy()
    else:
        # Filter by same m2 category AND m3 is null/empty
        same_category_df = df[
            (df['m2'] == query_m2) & 
            (df['m3'].isna() | (df['m3'].astype(str).str.strip() == ''))
        ].copy()
    
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
    columns_to_include = ['item', 'item_model', 'brand', 'category', 'item_name', 'product_url', 'bcat1', 'bcat2', 'm1', 'm2', 'm3', 'price']
    
    # First, collect top_k results by similarity
    for idx in top_indices:
        item_id = item_indices[idx]
        if item_id != query_item_id:  # Exclude query item
            similarity = similarities[idx]
            
            # Filter by similarity range
            if similarity < min_similarity or similarity > max_similarity:
                continue
            
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
            
            # Calculate price difference if both prices exist
            item_price = parse_price(result.get('price'))
            if query_price is not None and item_price is not None:
                result['price_diff'] = abs(item_price - query_price)
                # Store parsed price for display
                result['price_parsed'] = item_price
            else:
                result['price_diff'] = float('inf')  # If no price, put at end
                result['price_parsed'] = None
            
            results.append(result)
            if len(results) >= top_k:
                break
    
    # Sort by price difference (closest to query item's price first)
    # Items with no price (price_diff = inf) will be sorted to the end
    # For items with same price difference, sort by similarity (highest first) as tiebreaker
    results.sort(key=lambda x: (
        x['price_diff'] if x['price_diff'] != float('inf') else float('inf'),
        -x['similarity']  # Negative for descending order (highest similarity first)
    ))
    
    return results

def parse_price(price_value):
    """
    Parse price string to float, handling comma separators and various formats
    """
    if price_value is None or pd.isna(price_value):
        return None
    
    # Convert to string if not already
    price_str = str(price_value).strip()
    
    if not price_str or price_str == '':
        return None
    
    try:
        # Remove commas (thousands separators) and any other non-numeric characters except decimal point
        # Remove commas, spaces, and currency symbols
        cleaned_price = price_str.replace(',', '').replace(' ', '').replace('$', '').replace('€', '').replace('£', '')
        
        # Convert to float
        return float(cleaned_price)
    except (ValueError, TypeError):
        return None

def format_result_title(result, index):
    """Format the expander title with similarity, price, and price difference"""
    # Format price and difference for display
    price_str = ""
    if result.get('price_parsed') is not None:
        price_str = f", Price: {result['price_parsed']:,.2f}"
        if result.get('price_diff') != float('inf'):
            price_str += f", Diff: {result['price_diff']:,.2f}"
    return f"#{index} {result['item_name']} (Similarity: {result['similarity']:.4f}{price_str})"

def calculate_all_similarities_in_category(m2_category, embeddings, item_mapping, df):
    """
    Calculate all pairwise similarities within an m2 category
    Returns a list of similarity scores (excluding self-similarities)
    """
    # Filter items by m2 category
    category_df = df[df['m2'] == m2_category].copy()
    
    if category_df.empty:
        return []
    
    # Get embeddings for items in category
    category_items = category_df['item'].values
    item_indices = []
    category_embeddings = []
    
    for item_id in category_items:
        if item_id in item_mapping:
            idx = item_mapping[item_id]
            item_indices.append(item_id)
            category_embeddings.append(embeddings[idx])
    
    if len(category_embeddings) < 2:
        return []
    
    category_embeddings = np.array(category_embeddings)
    
    # Calculate all pairwise similarities
    similarity_matrix = cosine_similarity(category_embeddings)
    
    # Extract upper triangle (excluding diagonal) to avoid duplicates and self-similarities
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[upper_triangle_indices]
    
    return similarities.tolist()

# Streamlit App
st.set_page_config(page_title="Item Similarity Search", layout="wide")

# Sidebar navigation
page = st.sidebar.selectbox("Select Page", ["Item Search", "Similarity Analysis"])

# Load data
df = load_data()

if page == "Item Search":
    st.title("🔍 Item Similarity Search")
    st.markdown("Search for similar items using Word2Vec and FastText embeddings (optimized for Georgian language)")

    # Search interface
    col1, col2 = st.columns([2, 1])

    with col1:
        search_input = st.text_input("Search by Item ID or Name", placeholder="Enter item ID (e.g., 11960) or item name")

    with col2:
        top_k = st.number_input("Number of results", min_value=1, max_value=50, value=15)

    # Similarity filter
    st.markdown("### Similarity Filter")
    col_filter1, col_filter2 = st.columns(2)

    with col_filter1:
        max_similarity = st.number_input(
            "Maximum Similarity", 
            min_value=0.0, 
            max_value=1.0, 
            value=1.0, 
            step=0.01,
            format="%.2f",
            help="Maximum similarity score to include (e.g., 0.95)"
        )

    with col_filter2:
        min_similarity = st.number_input(
            "Minimum Similarity", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.0, 
            step=0.01,
            format="%.2f",
            help="Minimum similarity score to include (e.g., 0.80)"
        )

    # Validate similarity range
    if min_similarity > max_similarity:
        st.warning("⚠️ Minimum similarity cannot be greater than maximum similarity. Please adjust the values.")
        min_similarity, max_similarity = max_similarity, min_similarity

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
            columns_to_show = ['item', 'item_model', 'brand', 'category', 'item_name', 'product_url', 'bcat1', 'bcat2', 'm1', 'm2', 'm3', 'price']
            
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
                if query_data.get('price'):
                    price_val = parse_price(query_data['price'])
                    if price_val is not None:
                        st.text(f"Price: {price_val:,.2f}")
                    else:
                        st.text(f"Price: {query_data['price']}")
                if query_data['product_url']:
                    st.markdown(f"**Product URL:**")
                    st.markdown(f"[Link]({query_data['product_url']})")
            
            # Load embeddings
            with st.spinner("Loading embeddings..."):
                w2v_embeddings, w2v_mapping = load_word2vec_embeddings()
                fasttext_embeddings, fasttext_mapping = load_fasttext_embeddings()
                w2v_embeddings_gi, w2v_mapping_gi = load_word2vec_embeddings_general_info()
                fasttext_embeddings_gi, fasttext_mapping_gi = load_fasttext_embeddings_general_info()
                try:
                    jina_embeddings, jina_mapping = load_jina_embeddings()
                    jina_embeddings_gi, jina_mapping_gi = load_jina_embeddings_general_info()
                    jina_available = True
                except FileNotFoundError:
                    jina_available = False
                    st.warning("⚠️ Jina embeddings not found. Please run generate_jina_embeddings_api.py first.")
            
            # Find similar items
            st.markdown("---")
            st.subheader("🔎 Similar Items")
            
            # Create 6 columns: Raw Data models first (col1-3), then General Info models (col4-6)
            # Order: Word2Vec (Raw), FastText (Raw), Jina (Raw), Word2Vec (GI), FastText (GI), Jina (GI)
            col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.markdown("#### Word2Vec (Raw Data)")
            with st.spinner("Finding similar items with Word2Vec (Raw Data)..."):
                w2v_results = find_similar_items(
                    query_item_id, w2v_embeddings, w2v_mapping, 
                    df, top_k=top_k, min_similarity=min_similarity, max_similarity=max_similarity
                )
            
            if w2v_results:
                for i, result in enumerate(w2v_results, 1):
                    with st.expander(format_result_title(result, i), expanded=False):
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
                            if result.get('price'):
                                # Use parsed price if available, otherwise parse it
                                price_val = result.get('price_parsed')
                                if price_val is None:
                                    price_val = parse_price(result.get('price'))
                                if price_val is not None:
                                    st.text(f"Price: {price_val:,.2f}")
                                else:
                                    st.text(f"Price: {result['price']}")
                            if result['product_url']:
                                st.markdown(f"**Product URL:**")
                                st.markdown(f"[Link]({result['product_url']})")
            else:
                st.info("No similar items found in the same category")
        
        with col2:
            st.markdown("#### FastText (Raw Data)")
            with st.spinner("Finding similar items with FastText (Raw Data)..."):
                fasttext_results = find_similar_items(
                    query_item_id, fasttext_embeddings, fasttext_mapping, 
                    df, top_k=top_k, min_similarity=min_similarity, max_similarity=max_similarity
                )
            
            if fasttext_results:
                for i, result in enumerate(fasttext_results, 1):
                    with st.expander(format_result_title(result, i), expanded=False):
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
                            if result.get('price'):
                                # Use parsed price if available, otherwise parse it
                                price_val = result.get('price_parsed')
                                if price_val is None:
                                    price_val = parse_price(result.get('price'))
                                if price_val is not None:
                                    st.text(f"Price: {price_val:,.2f}")
                                else:
                                    st.text(f"Price: {result['price']}")
                            if result['product_url']:
                                st.markdown(f"**Product URL:**")
                                st.markdown(f"[Link]({result['product_url']})")
            else:
                st.info("No similar items found in the same category")
        
        with col3:
            if jina_available:
                st.markdown("#### Jina (Raw Data)")
                with st.spinner("Finding similar items with Jina (Raw Data)..."):
                    jina_results = find_similar_items(
                        query_item_id, jina_embeddings, jina_mapping, 
                        df, top_k=top_k, min_similarity=min_similarity, max_similarity=max_similarity
                    )
                
                if jina_results:
                    for i, result in enumerate(jina_results, 1):
                        with st.expander(format_result_title(result, i), expanded=False):
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
                                if result.get('price'):
                                    price_val = result.get('price_parsed')
                                    if price_val is None:
                                        price_val = parse_price(result.get('price'))
                                    if price_val is not None:
                                        st.text(f"Price: {price_val:,.2f}")
                                    else:
                                        st.text(f"Price: {result['price']}")
                                if result['product_url']:
                                    st.markdown(f"**Product URL:**")
                                    st.markdown(f"[Link]({result['product_url']})")
                else:
                    st.info("No similar items found in the same category")
            else:
                st.info("Jina embeddings not available")
        
        with col6:
            if jina_available:
                st.markdown("#### Jina (General Info)")
                with st.spinner("Finding similar items with Jina (General Info)..."):
                    jina_results_gi = find_similar_items(
                        query_item_id, jina_embeddings_gi, jina_mapping_gi, 
                        df, top_k=top_k, min_similarity=min_similarity, max_similarity=max_similarity
                    )
                
                if jina_results_gi:
                    for i, result in enumerate(jina_results_gi, 1):
                        with st.expander(format_result_title(result, i), expanded=False):
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
                                if result.get('price'):
                                    price_val = result.get('price_parsed')
                                    if price_val is None:
                                        price_val = parse_price(result.get('price'))
                                    if price_val is not None:
                                        st.text(f"Price: {price_val:,.2f}")
                                    else:
                                        st.text(f"Price: {result['price']}")
                                if result['product_url']:
                                    st.markdown(f"**Product URL:**")
                                    st.markdown(f"[Link]({result['product_url']})")
                else:
                    st.info("No similar items found in the same category")
            else:
                st.info("Jina embeddings not available")
        
        with col4:
            st.markdown("#### Word2Vec (General Info)")
            with st.spinner("Finding similar items with Word2Vec (General Info)..."):
                w2v_results_gi = find_similar_items(
                    query_item_id, w2v_embeddings_gi, w2v_mapping_gi, 
                    df, top_k=top_k, min_similarity=min_similarity, max_similarity=max_similarity
                )
            
            if w2v_results_gi:
                for i, result in enumerate(w2v_results_gi, 1):
                    with st.expander(format_result_title(result, i), expanded=False):
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
                            if result.get('price'):
                                price_val = result.get('price_parsed')
                                if price_val is None:
                                    price_val = parse_price(result.get('price'))
                                if price_val is not None:
                                    st.text(f"Price: {price_val:,.2f}")
                                else:
                                    st.text(f"Price: {result['price']}")
                            if result['product_url']:
                                st.markdown(f"**Product URL:**")
                                st.markdown(f"[Link]({result['product_url']})")
            else:
                st.info("No similar items found in the same category")
        
        with col5:
            st.markdown("#### FastText (General Info)")
            with st.spinner("Finding similar items with FastText (General Info)..."):
                fasttext_results_gi = find_similar_items(
                    query_item_id, fasttext_embeddings_gi, fasttext_mapping_gi, 
                    df, top_k=top_k, min_similarity=min_similarity, max_similarity=max_similarity
                )
            
            if fasttext_results_gi:
                for i, result in enumerate(fasttext_results_gi, 1):
                    with st.expander(format_result_title(result, i), expanded=False):
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
                            if result.get('price'):
                                price_val = result.get('price_parsed')
                                if price_val is None:
                                    price_val = parse_price(result.get('price'))
                                if price_val is not None:
                                    st.text(f"Price: {price_val:,.2f}")
                                else:
                                    st.text(f"Price: {result['price']}")
                            if result['product_url']:
                                st.markdown(f"**Product URL:**")
                                st.markdown(f"[Link]({result['product_url']})")
            else:
                st.info("No similar items found in the same category")
        
        with col6:
            if jina_available:
                st.markdown("#### Jina (General Info)")
                with st.spinner("Finding similar items with Jina (General Info)..."):
                    jina_results_gi = find_similar_items(
                        query_item_id, jina_embeddings_gi, jina_mapping_gi, 
                        df, top_k=top_k, min_similarity=min_similarity, max_similarity=max_similarity
                    )
                
                if jina_results_gi:
                    for i, result in enumerate(jina_results_gi, 1):
                        with st.expander(format_result_title(result, i), expanded=False):
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
                                if result.get('price'):
                                    price_val = result.get('price_parsed')
                                    if price_val is None:
                                        price_val = parse_price(result.get('price'))
                                    if price_val is not None:
                                        st.text(f"Price: {price_val:,.2f}")
                                    else:
                                        st.text(f"Price: {result['price']}")
                                if result['product_url']:
                                    st.markdown(f"**Product URL:**")
                                    st.markdown(f"[Link]({result['product_url']})")
                else:
                    st.info("No similar items found in the same category")
            else:
                st.info("Jina embeddings not available")
    else:
        st.info("👆 Enter an item ID or name to search for similar items")
        st.markdown("### Sample Items:")
        sample_items = df.head(10)[['item', 'item_name', 'brand', 'category']]
        st.dataframe(sample_items, width='stretch')

elif page == "Similarity Analysis":
    st.title("📊 Similarity Score Analysis")
    st.markdown("Analyze similarity score distributions by M2 category")
    
    # Load embeddings
    with st.spinner("Loading embeddings..."):
        w2v_embeddings, w2v_mapping = load_word2vec_embeddings()
        fasttext_embeddings, fasttext_mapping = load_fasttext_embeddings()
        w2v_embeddings_gi, w2v_mapping_gi = load_word2vec_embeddings_general_info()
        fasttext_embeddings_gi, fasttext_mapping_gi = load_fasttext_embeddings_general_info()
        try:
            jina_embeddings, jina_mapping = load_jina_embeddings()
            jina_embeddings_gi, jina_mapping_gi = load_jina_embeddings_general_info()
            jina_available = True
        except FileNotFoundError:
            jina_available = False
            st.warning("⚠️ Jina embeddings not found. Similarity analysis will only show Word2Vec and FastText.")
    
    # Get unique m2 categories with item counts
    unique_m2 = sorted(df['m2'].dropna().unique().tolist())
    
    if not unique_m2:
        st.error("No M2 categories found in the data")
    else:
        # Calculate item counts for each category
        m2_counts = df['m2'].value_counts().to_dict()
        
        # Create a table of all M2 categories with item counts
        st.markdown("### M2 Categories Overview")
        m2_table_data = {
            'M2 Category': unique_m2,
            'Number of Items': [m2_counts.get(m2, 0) for m2 in unique_m2]
        }
        m2_df = pd.DataFrame(m2_table_data)
        m2_df = m2_df.sort_values('Number of Items', ascending=False)
        st.dataframe(m2_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Create display names with counts
        m2_options = [f"{m2} ({m2_counts.get(m2, 0):,} items)" for m2 in unique_m2]
        
        # Category selection
        selected_m2_idx = st.selectbox(
            "Select M2 Category", 
            range(len(unique_m2)),
            format_func=lambda x: m2_options[x]
        )
        selected_m2 = unique_m2[selected_m2_idx]
        category_item_count = m2_counts.get(selected_m2, 0)
        
        # Display category info
        st.info(f"📦 Selected category contains **{category_item_count:,} items**. This will generate **{category_item_count * (category_item_count - 1) // 2:,}** pairwise similarity calculations.")
        
        # Threshold setting - default to category item count (but at least min_value)
        st.markdown("### Performance Threshold")
        threshold = st.number_input(
            "Maximum items for similarity calculation",
            min_value=10,
            max_value=10000,
            value=max(10, category_item_count),
            step=50,
            help="If category has more items than this threshold, it's recommended to retrieve all items instead of calculating similarities"
        )
        
        # Show recommendation
        if category_item_count > threshold:
            st.warning(f"⚠️ **Recommendation**: This category has {category_item_count:,} items (exceeds threshold of {threshold:,}). Consider retrieving all items instead of calculating similarities for better performance.")
            st.markdown(f"**Estimated calculation time**: ~{category_item_count * (category_item_count - 1) / 2 / 100000:.1f} seconds (rough estimate)**")
        else:
            st.success(f"✅ Category size ({category_item_count:,} items) is within threshold. Similarity calculation is feasible.")
        
        # Bin size input
        bin_size = st.number_input(
            "Bin Size",
            min_value=0.01,
            max_value=1.0,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="Size of each bin in the histogram (between 0 and 1)"
        )
        
        # Calculate similarities button
        if st.button("Generate Histograms"):
            import time
            start_time = time.time()
            
            with st.spinner(f"Calculating similarities for {category_item_count:,} items... This may take a while for large categories."):
                # Calculate similarities for each model
                w2v_start = time.time()
                w2v_similarities = calculate_all_similarities_in_category(
                    selected_m2, w2v_embeddings, w2v_mapping, df
                )
                w2v_time = time.time() - w2v_start
                
                fasttext_start = time.time()
                fasttext_similarities = calculate_all_similarities_in_category(
                    selected_m2, fasttext_embeddings, fasttext_mapping, df
                )
                fasttext_time = time.time() - fasttext_start
                
                w2v_gi_start = time.time()
                w2v_gi_similarities = calculate_all_similarities_in_category(
                    selected_m2, w2v_embeddings_gi, w2v_mapping_gi, df
                )
                w2v_gi_time = time.time() - w2v_gi_start
                
                fasttext_gi_start = time.time()
                fasttext_gi_similarities = calculate_all_similarities_in_category(
                    selected_m2, fasttext_embeddings_gi, fasttext_mapping_gi, df
                )
                fasttext_gi_time = time.time() - fasttext_gi_start
                
                # Calculate Jina similarities if available
                if jina_available:
                    jina_start = time.time()
                    jina_similarities = calculate_all_similarities_in_category(
                        selected_m2, jina_embeddings, jina_mapping, df
                    )
                    jina_time = time.time() - jina_start
                    
                    jina_gi_start = time.time()
                    jina_gi_similarities = calculate_all_similarities_in_category(
                        selected_m2, jina_embeddings_gi, jina_mapping_gi, df
                    )
                    jina_gi_time = time.time() - jina_gi_start
                else:
                    jina_similarities = []
                    jina_gi_similarities = []
                    jina_time = 0
                    jina_gi_time = 0
            
            total_time = time.time() - start_time
            
            # Display performance metrics
            st.success(f"✅ Calculation completed in {total_time:.2f} seconds")
            if jina_available:
                col_perf1, col_perf2, col_perf3, col_perf4, col_perf5, col_perf6 = st.columns(6)
            else:
                col_perf1, col_perf2, col_perf3, col_perf4, col_perf5, col_perf6 = st.columns(6)
            # Performance metrics: Raw Data first, then General Info
            with col_perf1:
                st.metric("Word2Vec (Raw)", f"{w2v_time:.2f}s", f"{len(w2v_similarities):,} pairs")
            with col_perf2:
                st.metric("FastText (Raw)", f"{fasttext_time:.2f}s", f"{len(fasttext_similarities):,} pairs")
            with col_perf3:
                if jina_available:
                    st.metric("Jina (Raw)", f"{jina_time:.2f}s", f"{len(jina_similarities):,} pairs")
                else:
                    st.metric("Jina (Raw)", "N/A", "Not available")
            with col_perf4:
                st.metric("Word2Vec (GI)", f"{w2v_gi_time:.2f}s", f"{len(w2v_gi_similarities):,} pairs")
            with col_perf5:
                st.metric("FastText (GI)", f"{fasttext_gi_time:.2f}s", f"{len(fasttext_gi_similarities):,} pairs")
            with col_perf6:
                if jina_available:
                    st.metric("Jina (GI)", f"{jina_gi_time:.2f}s", f"{len(jina_gi_similarities):,} pairs")
                else:
                    st.metric("Jina (GI)", "N/A", "Not available")
            
            # Create histograms
            if w2v_similarities or fasttext_similarities or w2v_gi_similarities or fasttext_gi_similarities or (jina_available and (jina_similarities or jina_gi_similarities)):
                st.markdown("---")
                st.subheader(f"Similarity Score Distributions for: {selected_m2}")
                
                # Create 6 columns for histograms
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    st.markdown("#### Word2Vec (Raw Data)")
                    if w2v_similarities:
                        fig = go.Figure(data=[go.Histogram(
                            x=w2v_similarities,
                            nbinsx=int(1/bin_size),
                            marker_color='#1f77b4',
                            marker_line_color='black',
                            marker_line_width=1,
                            opacity=0.7
                        )])
                        fig.update_layout(
                            title='Word2Vec (Raw Data)',
                            xaxis_title='Similarity Score',
                            yaxis_title='Frequency',
                            xaxis=dict(range=[0, 1]),
                            height=400,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"Similarity pairs: {len(w2v_similarities):,} | Mean: {np.mean(w2v_similarities):.4f} | Std: {np.std(w2v_similarities):.4f}")
                    else:
                        st.info("No similarities calculated")
                
                with col2:
                    st.markdown("#### FastText (Raw Data)")
                    if fasttext_similarities:
                        fig = go.Figure(data=[go.Histogram(
                            x=fasttext_similarities,
                            nbinsx=int(1/bin_size),
                            marker_color='#ff7f0e',
                            marker_line_color='black',
                            marker_line_width=1,
                            opacity=0.7
                        )])
                        fig.update_layout(
                            title='FastText (Raw Data)',
                            xaxis_title='Similarity Score',
                            yaxis_title='Frequency',
                            xaxis=dict(range=[0, 1]),
                            height=400,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"Similarity pairs: {len(fasttext_similarities):,} | Mean: {np.mean(fasttext_similarities):.4f} | Std: {np.std(fasttext_similarities):.4f}")
                    else:
                        st.info("No similarities calculated")
                
                with col3:
                    st.markdown("#### Jina (Raw Data)")
                    if jina_available and jina_similarities:
                        fig = go.Figure(data=[go.Histogram(
                            x=jina_similarities,
                            nbinsx=int(1/bin_size),
                            marker_color='#9467bd',
                            marker_line_color='black',
                            marker_line_width=1,
                            opacity=0.7
                        )])
                        fig.update_layout(
                            title='Jina (Raw Data)',
                            xaxis_title='Similarity Score',
                            yaxis_title='Frequency',
                            xaxis=dict(range=[0, 1]),
                            height=400,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"Similarity pairs: {len(jina_similarities):,} | Mean: {np.mean(jina_similarities):.4f} | Std: {np.std(jina_similarities):.4f}")
                    else:
                        st.info("No similarities calculated" if jina_available else "Jina embeddings not available")
                
                with col4:
                    st.markdown("#### Word2Vec (General Info)")
                    if w2v_gi_similarities:
                        fig = go.Figure(data=[go.Histogram(
                            x=w2v_gi_similarities,
                            nbinsx=int(1/bin_size),
                            marker_color='#2ca02c',
                            marker_line_color='black',
                            marker_line_width=1,
                            opacity=0.7
                        )])
                        fig.update_layout(
                            title='Word2Vec (General Info)',
                            xaxis_title='Similarity Score',
                            yaxis_title='Frequency',
                            xaxis=dict(range=[0, 1]),
                            height=400,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"Similarity pairs: {len(w2v_gi_similarities):,} | Mean: {np.mean(w2v_gi_similarities):.4f} | Std: {np.std(w2v_gi_similarities):.4f}")
                    else:
                        st.info("No similarities calculated")
                
                with col4:
                    st.markdown("#### Word2Vec (General Info)")
                    if w2v_gi_similarities:
                        fig = go.Figure(data=[go.Histogram(
                            x=w2v_gi_similarities,
                            nbinsx=int(1/bin_size),
                            marker_color='#2ca02c',
                            marker_line_color='black',
                            marker_line_width=1,
                            opacity=0.7
                        )])
                        fig.update_layout(
                            title='Word2Vec (General Info)',
                            xaxis_title='Similarity Score',
                            yaxis_title='Frequency',
                            xaxis=dict(range=[0, 1]),
                            height=400,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"Similarity pairs: {len(w2v_gi_similarities):,} | Mean: {np.mean(w2v_gi_similarities):.4f} | Std: {np.std(w2v_gi_similarities):.4f}")
                    else:
                        st.info("No similarities calculated")
                
                with col5:
                    st.markdown("#### FastText (General Info)")
                    if fasttext_gi_similarities:
                        fig = go.Figure(data=[go.Histogram(
                            x=fasttext_gi_similarities,
                            nbinsx=int(1/bin_size),
                            marker_color='#d62728',
                            marker_line_color='black',
                            marker_line_width=1,
                            opacity=0.7
                        )])
                        fig.update_layout(
                            title='FastText (General Info)',
                            xaxis_title='Similarity Score',
                            yaxis_title='Frequency',
                            xaxis=dict(range=[0, 1]),
                            height=400,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"Similarity pairs: {len(fasttext_gi_similarities):,} | Mean: {np.mean(fasttext_gi_similarities):.4f} | Std: {np.std(fasttext_gi_similarities):.4f}")
                    else:
                        st.info("No similarities calculated")
                
                with col6:
                    st.markdown("#### Jina (General Info)")
                    if jina_available and jina_gi_similarities:
                        fig = go.Figure(data=[go.Histogram(
                            x=jina_gi_similarities,
                            nbinsx=int(1/bin_size),
                            marker_color='#8c564b',
                            marker_line_color='black',
                            marker_line_width=1,
                            opacity=0.7
                        )])
                        fig.update_layout(
                            title='Jina (General Info)',
                            xaxis_title='Similarity Score',
                            yaxis_title='Frequency',
                            xaxis=dict(range=[0, 1]),
                            height=400,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"Similarity pairs: {len(jina_gi_similarities):,} | Mean: {np.mean(jina_gi_similarities):.4f} | Std: {np.std(jina_gi_similarities):.4f}")
                    else:
                        st.info("No similarities calculated" if jina_available else "Jina embeddings not available")
            else:
                st.warning(f"No items found in category '{selected_m2}' or insufficient data for similarity calculation")

