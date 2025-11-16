"""
Beauty data loading utilities
"""

import json
import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
from models.beauty_model import load_beauty_checkpoint


@st.cache_resource
def load_beauty_artifacts(data_dir, device):
    """
    Load all Beauty artifacts: items, mappings, and trained model
    """
    data_path = Path(data_dir)
    
    # Load items catalog
    items_df = pd.read_parquet(data_path / "items.parquet")
    
    # Load ASIN mappings
    with open(data_path / "mappings.json", "r") as f:
        maps = json.load(f)
    item2idx = maps["item2idx"]
    idx2item = {v: k for k, v in item2idx.items()}
    
    # Load model checkpoint (contains model + embeddings)
    checkpoint_path = data_path / "checkpoints" / "model_checkpoint.pt"
    checkpoint_data = load_beauty_checkpoint(checkpoint_path, device)
    
    return {
        "items_df": items_df,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "checkpoint_data": checkpoint_data,
        "device": device,
    }


def clean_product_text(text):
    """Remove metadata tags from product text"""
    import re
    cleaned = re.sub(r'\[([A-Z_]+)\]', '', text)
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()


def search_beauty_items(query_text, artifacts, top_n=10):
    """
    Search beauty items using text similarity
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Load encoder (cached)
    if 'beauty_encoder' not in st.session_state:
        st.session_state.beauty_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    encoder = st.session_state.beauty_encoder
    query_emb = encoder.encode([query_text])[0]
    
    # Use checkpoint embeddings for search
    item_embs = artifacts['checkpoint_data']['item_embs']
    sims = cosine_similarity(query_emb.reshape(1, -1), item_embs)[0]
    
    top_idxs = np.argsort(sims)[::-1][:top_n]
    
    results = []
    for idx in top_idxs:
        asin = artifacts["idx2item"][idx]
        item_text = artifacts["items_df"].iloc[idx]["item_text"]
        results.append({
            "asin": asin,
            "item_text": item_text,
            "similarity": sims[idx]
        })
    
    return results