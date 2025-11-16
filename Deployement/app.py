"""
Multi-Product Neural Retrieval Recommender System
Supports: Electronics (MLP Projector) and Beauty (Two-Tower)
Beautiful, Modern UI with Dark/Light Mode Toggle
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# Import product configurations
from config.product_config import PRODUCTS, UI_CONFIG

# Import Beauty model components
from models.beauty_model import get_user_embedding_beauty, recommend_top_k_beauty
from utils.beauty_loader import load_beauty_artifacts, clean_product_text, search_beauty_items

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Product Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)


if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'


if st.session_state.theme == 'light':
    st.markdown("""
    <style>
        /* Light Mode - Clean & Professional */
        .stApp {
            background: linear-gradient(135deg, #edf2ff 0%, #e0f4ff 40%, #fef6ff 100%);
            color: #1a202c;
        }

        /* Global text color for readability */
        body, .main, .main .block-container, .stMarkdown, .stText, label, span, p {
            color: #1a202c;
        }
        
        .main .block-container {
            padding: 2rem 3rem;
            max-width: 1400px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
            margin: 2.5rem auto;
            border: 1px solid rgba(148, 163, 184, 0.35);
            backdrop-filter: blur(18px);
        }
        
        /* Headers */
        .app-title {
            /* Light theme: cool blue-purple gradient */
            background: linear-gradient(120deg, #1d4ed8, #9333ea, #db2777);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            color: transparent !important;
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.5rem !important;
            letter-spacing: -0.5px;
            text-shadow: 0 6px 18px rgba(148, 163, 184, 0.55);
        }
        
        h3 {
            color: #2d3748;
            font-weight: 600;
            font-size: 1.5rem;
            margin-top: 2rem;
        }
        
        .subtitle {
            color: #718096;
            font-size: 1rem;
            margin-bottom: 2rem;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: linear-gradient(120deg, rgba(96, 165, 250, 0.06), rgba(79, 70, 229, 0.06));
            padding: 0.25rem;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: transparent;
            border-radius: 8px;
            color: #4a5568;
            font-weight: 500;
            padding: 0 2rem;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(226, 232, 240, 0.9);
            color: #1f2933;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #6366f1, #ec4899);
            color: white !important;
            box-shadow: 0 10px 25px rgba(79, 70, 229, 0.4);
        }
        
        /* Buttons - clean, minimal with accent hover */
        .stButton > button {
            background: #ffffff;
            color: #1f2933;
            border-radius: 999px;
            border: 1px solid #e5e7eb;
            padding: 0.6rem 1.7rem;
            font-weight: 500;
            letter-spacing: 0.01em;
            box-shadow: 0 4px 10px rgba(15, 23, 42, 0.08);
            transition: all 0.18s ease-out;
        }
        
        .stButton > button:hover {
            background: linear-gradient(120deg, #6366f1, #ec4899);
            color: #ffffff;
            border-color: transparent;
            box-shadow: 0 10px 24px rgba(148, 27, 81, 0.28);
            transform: translateY(-1px);
        }
        
        .stButton > button:disabled {
            background: #cbd5e0;
            color: #a0aec0;
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        
        /* Text inputs */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }

        /* Containers */
        div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stVerticalBlock"]) {
            border-radius: 8px;
            padding: 1.5rem;
            background: linear-gradient(135deg, #ffffff, #f1f5f9);
            border: 1px solid rgba(148, 163, 184, 0.4);
            transition: all 0.2s;
        }
        
        div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stVerticalBlock"]):hover {
            box-shadow: 0 16px 35px rgba(15, 23, 42, 0.12);
            transform: translateY(-2px);
            border-color: #6366f1;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(135deg, #ffffff, #e5edff);
            border-right: 1px solid rgba(148, 163, 184, 0.4);
        }
        
        section[data-testid="stSidebar"] h3 {
            color: #2d3748;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }
        
        div[data-testid="stDataFrame"] > div {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        
        div[data-testid="stDataFrame"] table {
            font-size: 0.95rem;
        }
        
        div[data-testid="stDataFrame"] thead tr th {
            background-color: #f7fafc !important;
            color: #2d3748 !important;
            font-weight: 600 !important;
            padding: 1rem !important;
            border-bottom: 2px solid #e2e8f0 !important;
        }
        
        div[data-testid="stDataFrame"] tbody tr td {
            padding: 0.75rem !important;
            border-bottom: 1px solid #f7fafc !important;
        }
        
        div[data-testid="stDataFrame"] tbody tr:hover {
            background-color: #f7fafc !important;
        }

        /* Info/Success/Warning */
        .stAlert {
            border-radius: 12px;
            border-left: 4px solid #6366f1;
            background: linear-gradient(120deg, rgba(79, 70, 229, 0.06), rgba(56, 189, 248, 0.06));
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
        }
        
        /* Caption */
        .stCaption {
            color: #718096;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Metric */
        div[data-testid="stMetricValue"] {
            color: #2d3748;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        /* Dark Mode - High-contrast, neon-on-black */
        .stApp {
            background:
                radial-gradient(circle at top, rgba(236, 72, 153, 0.22), transparent 55%),
                radial-gradient(circle at bottom, rgba(56, 189, 248, 0.18), transparent 55%),
                #020617;
            color: #e5e7eb;
        }

        /* Global text color for readability */
        body, .main, .main .block-container, .stMarkdown, .stText, label, span, p {
            color: #e5e7eb;
        }
        
        .main .block-container {
            padding: 2rem 3rem;
            max-width: 1400px;
            background: rgba(5, 8, 22, 0.96);
            border-radius: 22px;
            box-shadow: 0 26px 70px rgba(0, 0, 0, 0.9);
            margin: 2.5rem auto;
            border: 1px solid rgba(148, 163, 184, 0.35);
            backdrop-filter: blur(20px);
        }
        
        /* Headers */
        .app-title {
            /* Dark theme: cyan-violet-orange neon gradient */
            background: linear-gradient(120deg, #22d3ee, #a855f7, #f97316);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            color: transparent !important;
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.5rem !important;
            letter-spacing: -0.5px;
            text-shadow: 0 14px 32px rgba(0, 0, 0, 0.95);
        }
        
        h3 {
            color: #e2e8f0;
            font-weight: 600;
            font-size: 1.5rem;
            margin-top: 2rem;
        }
        
        .subtitle {
            color: #a0aec0;
            font-size: 1rem;
            margin-bottom: 2rem;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(15, 23, 42, 0.9);
            padding: 0.25rem;
            border-radius: 999px;
            border: 1px solid rgba(75, 85, 99, 0.9);
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            background-color: transparent;
            border-radius: 999px;
            color: #9ca3af;
            font-weight: 500;
            padding: 0 2rem;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(31, 41, 55, 0.9);
            color: #f9fafb;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #f97316, #ec4899, #6366f1);
            color: #0b1020 !important;
            box-shadow: 0 12px 32px rgba(236, 72, 153, 0.45);
        }
        
        /* Buttons - subtle outline with neon hover */
        .stButton > button {
            background: #020617;
            color: #f9fafb;
            border-radius: 999px;
            border: 1px solid #4b5563;
            padding: 0.6rem 1.7rem;
            font-weight: 500;
            letter-spacing: 0.01em;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.8);
            transition: all 0.18s ease-out;
        }
        
        .stButton > button:hover {
            background: linear-gradient(120deg, #6366f1, #ec4899);
            border-color: transparent;
            box-shadow: 0 14px 32px rgba(0, 0, 0, 0.95);
            transform: translateY(-1px);
        }
        
        .stButton > button:disabled {
            background: #4a5568;
            color: #718096;
        }
        
        /* Select boxes */
        .stSelectbox > div > div {
            border-radius: 8px;
            border: 1px solid #4a5568;
            background: #2d3748;
            color: #e2e8f0;
        }
        
        /* Text inputs */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            border-radius: 8px;
            border: 1px solid #4a5568;
            background: #2d3748;
            color: #e2e8f0;
        }
        
        /* Containers */
        div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stVerticalBlock"]) {
            border-radius: 14px;
            padding: 1.5rem;
            background: radial-gradient(circle at top left, rgba(15, 23, 42, 0.98), rgba(3, 7, 18, 0.98));
            border: 1px solid rgba(55, 65, 81, 0.9);
            transition: all 0.2s;
        }
        
        div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stVerticalBlock"]):hover {
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.95);
            border-color: rgba(236, 72, 153, 0.6);
            transform: translateY(-2px);
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617, #020617 40%, #030712 100%);
            border-right: 1px solid rgba(31, 41, 55, 0.95);
        }
        
        section[data-testid="stSidebar"] h3 {
            color: #e2e8f0;
        }
        
        section[data-testid="stSidebar"] .stMarkdown {
            color: #cbd5e0;
        }
        
        /* Dataframe styling - Dark Mode */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }
        
        div[data-testid="stDataFrame"] > div {
            border-radius: 10px;
            border: 1px solid rgba(55, 65, 81, 0.9);
        }
        
        div[data-testid="stDataFrame"] table {
            font-size: 0.95rem;
        }
        
        div[data-testid="stDataFrame"] thead tr th {
            background-color: #020617 !important;
            color: #e5e7eb !important;
            font-weight: 600 !important;
            padding: 1rem !important;
            border-bottom: 2px solid rgba(55, 65, 81, 0.9) !important;
        }
        
        div[data-testid="stDataFrame"] tbody tr td {
            padding: 0.75rem !important;
            border-bottom: 1px solid #020617 !important;
            color: #e5e7eb !important;
        }
        
        div[data-testid="stDataFrame"] tbody tr:hover {
            background-color: #020617 !important;
        }

        /* Caption */
        .stCaption {
            color: #a0aec0;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Metric */
        div[data-testid="stMetricValue"] {
            color: #e2e8f0;
            text-shadow: 0 2px 6px rgba(15, 23, 42, 0.9);
        }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# ELECTRONICS MODEL 
# ==========================================
TEXT_EMB_DIM = 384
EMB_D = 128

class ItemProjector(nn.Module):
    def __init__(self, in_dim=TEXT_EMB_DIM, out_dim=EMB_D, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        z = self.mlp(x)
        z = F.normalize(z, dim=-1)
        return z

@st.cache_resource
def load_electronics_artifacts(data_dir):
    """Load Electronics model artifacts"""
    data_path = Path(data_dir)
    artifacts_path = data_path / "optionB_nlp_model"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    items_df = pd.read_parquet(data_path / "items.parquet")
    
    with open(data_path / "mappings.json", "r") as f:
        maps = json.load(f)
    item2idx = maps["item2idx"]
    idx2item = {v: k for k, v in item2idx.items()}
    
    base_text_emb = np.load(data_path / "item_text_emb_384.npy")
    base_text_emb_t = torch.from_numpy(base_text_emb).to(device=device, dtype=torch.float32)
    
    projector = ItemProjector().to(device)
    projector.load_state_dict(
        torch.load(artifacts_path / "item_projector.pt", map_location=device, weights_only=True)
    )
    projector.eval()
    
    proj_emb_path = artifacts_path / "E_item_proj_128.npy"
    if proj_emb_path.exists():
        E_item_proj = np.load(proj_emb_path)
    else:
        with torch.no_grad():
            E_item_proj_t = projector(base_text_emb_t)
            E_item_proj = E_item_proj_t.cpu().numpy()
        np.save(proj_emb_path, E_item_proj)
    
    return {
        "items_df": items_df,
        "item2idx": item2idx,
        "idx2item": idx2item,
        "base_text_emb": base_text_emb,
        "base_text_emb_t": base_text_emb_t,
        "projector": projector,
        "E_item_proj": E_item_proj,
        "device": device,
    }

def get_user_embedding_electronics(history_asins, artifacts):
    """Generate user embedding for Electronics"""
    item2idx = artifacts["item2idx"]
    projector = artifacts["projector"]
    base_text_emb_t = artifacts["base_text_emb_t"]
    device = artifacts["device"]
    
    history_idxs = []
    invalid_asins = []
    
    for asin in history_asins:
        if asin in item2idx:
            history_idxs.append(item2idx[asin])
        else:
            invalid_asins.append(asin)
    
    if len(history_idxs) == 0:
        return None, invalid_asins
    
    hist_idx_t = torch.tensor(history_idxs, dtype=torch.long, device=device)
    hist_base = base_text_emb_t[hist_idx_t]
    
    with torch.no_grad():
        hist_proj = projector(hist_base)
    
    user_emb = hist_proj.mean(dim=0)
    user_emb = F.normalize(user_emb, dim=0)
    
    return user_emb.cpu().numpy(), invalid_asins

def recommend_top_k_electronics(user_emb, E_item_proj, k=10, exclude_idxs=None):
    """Get top-k recommendations for Electronics"""
    sims = cosine_similarity(user_emb.reshape(1, -1), E_item_proj)[0]
    
    if exclude_idxs is not None:
        sims[exclude_idxs] = -1.0
    
    topk_idxs = np.argsort(sims)[::-1][:k]
    topk_scores = sims[topk_idxs]
    
    return topk_idxs, topk_scores

def search_electronics_items(query_text, artifacts, top_n=10):
    """Search electronics items"""
    from sentence_transformers import SentenceTransformer
    
    if 'electronics_encoder' not in st.session_state:
        st.session_state.electronics_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    encoder = st.session_state.electronics_encoder
    query_emb = encoder.encode([query_text])[0]
    
    base_text_emb = artifacts["base_text_emb"]
    sims = cosine_similarity(query_emb.reshape(1, -1), base_text_emb)[0]
    
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

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def get_category_icon(item_text, product_type='Electronics'):
    """Get emoji icon based on product category"""
    item_lower = item_text.lower()
    
    if product_type == 'Electronics':
        if 'gaming' in item_lower or 'game' in item_lower or 'controller' in item_lower:
            return '🎮'
        elif 'storage' in item_lower or 'ssd' in item_lower or 'hard drive' in item_lower:
            return '💾'
        elif 'keyboard' in item_lower:
            return '⌨️'
        elif 'mouse' in item_lower or 'mice' in item_lower:
            return '🖱️'
        elif 'camera' in item_lower or 'lens' in item_lower or 'webcam' in item_lower:
            return '📷'
        elif 'headphone' in item_lower or 'audio' in item_lower or 'speaker' in item_lower or 'earbud' in item_lower:
            return '🎧'
        elif 'phone' in item_lower or 'mobile' in item_lower or 'smartphone' in item_lower:
            return '📱'
        elif 'cable' in item_lower or 'wire' in item_lower or 'usb' in item_lower or 'hdmi' in item_lower:
            return '🔌'
        elif 'monitor' in item_lower or 'screen' in item_lower or 'display' in item_lower:
            return '🖥️'
        elif 'laptop' in item_lower or 'notebook' in item_lower or 'macbook' in item_lower:
            return '💻'
        else:
            return '📦'
    else:  # Beauty
        if 'skin' in item_lower or 'moistur' in item_lower or 'cream' in item_lower:
            return '🧴'
        elif 'makeup' in item_lower or 'lipstick' in item_lower or 'foundation' in item_lower:
            return '💄'
        elif 'hair' in item_lower or 'shampoo' in item_lower or 'conditioner' in item_lower:
            return '💇'
        elif 'perfume' in item_lower or 'fragrance' in item_lower or 'cologne' in item_lower:
            return '🌸'
        elif 'brush' in item_lower or 'tool' in item_lower:
            return '🖌️'
        elif 'nail' in item_lower or 'polish' in item_lower:
            return '💅'
        elif 'body' in item_lower or 'lotion' in item_lower or 'bath' in item_lower:
            return '🛁'
        elif 'health' in item_lower or 'wellness' in item_lower:
            return '💊'
        else:
            return '✨'

# MAIN APP
def main():
    # Initialize product selection
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = 'Electronics'
    
    # Theme toggle button (top right)
    col_theme1, col_theme2 = st.columns([6, 1])
    with col_theme2:
        theme_icon = "☀️" if st.session_state.theme == 'dark' else "🌙"
        if st.button(theme_icon, key="theme_toggle", help="Toggle theme"):
            toggle_theme()
            st.rerun()
    
    # Product selector (prominent)
    col_title1, col_title2 = st.columns([3, 1])
    
    with col_title1:
        st.markdown("<h1 class='app-title'>Smart Product Recommender</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Discover personalized recommendations powered by neural retrieval</p>", unsafe_allow_html=True)
    
    with col_title2:
        product_options = list(PRODUCTS.keys())
        selected_product = st.selectbox(
            "Category",
            product_options,
            index=product_options.index(st.session_state.selected_product),
            key="product_selector",
            label_visibility="collapsed"
        )
        
        # Update session state if changed
        if selected_product != st.session_state.selected_product:
            st.session_state.selected_product = selected_product
            # Clear cart when switching products
            st.session_state.cart = []
            st.rerun()
    
    st.markdown("---")
    
    # Get product config
    product_config = PRODUCTS[selected_product]
    product_icon = product_config['icon']
    
    # Display selected product
    st.markdown(f"### {product_icon} {product_config['name']} Recommendations")
    
    # Load appropriate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        if product_config['model_type'] == 'projector':
            # Load Electronics model
            artifacts = load_electronics_artifacts(product_config['data_dir'])
        else:
            # Load Beauty model
            artifacts = load_beauty_artifacts(product_config['data_dir'], device)
    except Exception as e:
        st.error(f"Error loading {product_config['name']} model: {str(e)}")
        st.stop()
    
    # Initialize cart
    if 'cart' not in st.session_state:
        st.session_state.cart = []
    
    # Cart count badge
    if st.session_state.cart:
        st.info(f"**{len(st.session_state.cart)} items** selected")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Browse Products", "Enter ASINs", "Search"])
    
    # TAB 1: BROWSE
    with tab1:
        st.markdown("### Discover Products")
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            categories = list(product_config['categories'].keys())
            category = st.selectbox(
                "Category",
                categories,
                label_visibility="collapsed"
            )
        
        category_key = product_config['categories'][category]
        
        with col2:
            items_per_page = st.selectbox(
                "Items", 
                UI_CONFIG['items_per_page_options'], 
                index=0,
                label_visibility="collapsed"
            )
        
        with col3:
            if st.button("Refresh", use_container_width=True):
                st.session_state.random_seed = st.session_state.get('random_seed', 0) + 1
                st.rerun()
        
        # Get products
        import random
        seed = st.session_state.get('random_seed', 42)
        random.seed(seed)
        all_indices = list(range(len(artifacts['items_df'])))
        random.shuffle(all_indices)
        display_indices = all_indices[:items_per_page * 3]
        
        if category_key != "All":
            filtered = []
            for idx in display_indices:
                item_text = artifacts['items_df'].iloc[idx]['item_text']
                if category_key.lower() in item_text.lower():
                    filtered.append(idx)
            display_indices = filtered[:items_per_page]
        else:
            display_indices = display_indices[:items_per_page]
        
        if not display_indices:
            st.warning(f"No products found for {category}")
        else:
            st.caption(f"Showing {len(display_indices)} products")
            
            # Product grid
            cols_per_row = UI_CONFIG['cols_per_row_browse']
            for i in range(0, len(display_indices), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for col_idx, col in enumerate(cols):
                    if i + col_idx < len(display_indices):
                        idx = display_indices[i + col_idx]
                        asin = artifacts['idx2item'][idx]
                        item_text = artifacts['items_df'].iloc[idx]['item_text']
                        
                        with col:
                            with st.container(border=True):
                                icon = get_category_icon(item_text, selected_product)
                                clean_text = clean_product_text(item_text)
                                
                                # Icon row
                                st.markdown(
                                    f"<div style='font-size: 2rem; text-align: center; margin-bottom: 0.5rem;'>{icon}</div>",
                                    unsafe_allow_html=True,
                                )
                                # Product title
                                st.markdown(f"**{clean_text[:70]}...**")
                                st.caption(f"`{asin}`")
                                
                                if asin not in st.session_state.cart:
                                    if st.button("Add", key=f"add_{asin}", use_container_width=True):
                                        st.session_state.cart.append(asin)
                                        st.rerun()
                                else:
                                    st.button("Added", key=f"in_{asin}", disabled=True, use_container_width=True)
    
    # TAB 2: MANUAL ENTRY
    with tab2:
        st.markdown("### Enter Product Codes")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            asins_input = st.text_area(
                "ASINs",
                height=300,
                placeholder="B00FSTW88K\nB009CQ01ZQ\nB00OQVZDJM",
                label_visibility="collapsed"
            )
            
            if asins_input:
                asins_input = asins_input.replace(",", "\n")
                manual_asins = [asin.strip() for asin in asins_input.split("\n") if asin.strip()]
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.success(f"{len(manual_asins)} ASINs ready")
                with col_b:
                    if st.button("Load to Selection", use_container_width=True):
                        st.session_state.cart = manual_asins
                        st.rerun()
        
        with col2:
            st.info("""
            **Format**
            
            One per line:
```
            B00FSTW88K
            B009CQ01ZQ
```
            
            Or comma-separated
            """)
    
    # TAB 3: SEARCH
    with tab3:
        st.markdown("### Search Products")
        
        search_placeholder = {
            'Electronics': "Try: gaming mouse, USB cable, wireless keyboard...",
            'Beauty': "Try: moisturizer, lipstick, shampoo, face mask..."
        }
        
        query = st.text_input(
            "Search",
            placeholder=search_placeholder.get(selected_product, "Search products..."),
            label_visibility="collapsed"
        )
        
        if query:
            # Use appropriate search function
            if product_config['model_type'] == 'projector':
                results = search_electronics_items(query, artifacts, top_n=12)
            else:
                results = search_beauty_items(query, artifacts, top_n=12)
            
            st.caption(f"Found {len(results)} results")
            
            cols_per_row = UI_CONFIG['cols_per_row_search']
            for i in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                
                for col_idx, col in enumerate(cols):
                    if i + col_idx < len(results):
                        res = results[i + col_idx]
                        
                        with col:
                            with st.container(border=True):
                                icon = get_category_icon(res['item_text'], selected_product)
                                clean_text = clean_product_text(res['item_text'])
                                
                                st.markdown(
                                    f"<div style='font-size: 2rem; text-align: center; margin-bottom: 0.5rem;'>{icon}</div>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown(f"**{clean_text[:80]}...**")
                                st.caption(f"`{res['asin']}` • {res['similarity']:.0%}")
                                
                                if res['asin'] not in st.session_state.cart:
                                    if st.button("Add", key=f"s_{res['asin']}", use_container_width=True):
                                        st.session_state.cart.append(res['asin'])
                                        st.rerun()
                                else:
                                    st.button("Added", key=f"si_{res['asin']}", disabled=True, use_container_width=True)
    
    # SIDEBAR: CART
    with st.sidebar:
        st.markdown("### Your Selection")
        
        if st.session_state.cart:
            st.metric("Items", len(st.session_state.cart))
            st.markdown("---")
            
            for idx, asin in enumerate(st.session_state.cart):
                if asin in artifacts['item2idx']:
                    item_idx = artifacts['item2idx'][asin]
                    item_text = artifacts['items_df'].iloc[item_idx]['item_text']
                    clean_text = clean_product_text(item_text)
                    icon = get_category_icon(item_text, selected_product)
                    item_name = clean_text[:32]
                    
                    col_a, col_b = st.columns([4, 1])
                    with col_a:
                        st.caption(f"{idx+1}. {icon} {item_name}...")
                    with col_b:
                        if st.button("×", key=f"rm_{idx}"):
                            st.session_state.cart.pop(idx)
                            st.rerun()
            
            st.markdown("---")
            if st.button("Clear All", use_container_width=True):
                st.session_state.cart = []
                st.rerun()
        else:
            st.info("Selection is empty.\n\nBrowse and add products to get started.")
    
    # GENERATE RECOMMENDATIONS
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("Generate Recommendations", use_container_width=True):
            if not st.session_state.cart:
                st.warning("Please add some products first")
            else:
                with st.spinner("Analyzing preferences..."):
                    # Generate recommendations based on product type
                    if product_config['model_type'] == 'projector':
                        # Electronics
                        user_emb, invalid = get_user_embedding_electronics(st.session_state.cart, artifacts)
                        
                        if invalid:
                            st.warning(f"Invalid ASINs: {', '.join(invalid[:3])}")
                        
                        if user_emb is None:
                            st.error("No valid products found")
                        else:
                            item2idx = artifacts["item2idx"]
                            history_idxs = [item2idx[asin] for asin in st.session_state.cart if asin in item2idx]
                            
                            topk_idxs, topk_scores = recommend_top_k_electronics(
                                user_emb,
                                artifacts["E_item_proj"],
                                k=UI_CONFIG['max_recommendations'],
                                exclude_idxs=history_idxs
                            )
                    else:
                        # Beauty
                        user_emb, invalid = get_user_embedding_beauty(st.session_state.cart, artifacts['checkpoint_data'])
                        
                        if invalid:
                            st.warning(f"Invalid ASINs: {', '.join(invalid[:3])}")
                        
                        if user_emb is None:
                            st.error("No valid products found")
                        else:
                            id2idx = artifacts['checkpoint_data']['id2idx']
                            history_idxs = [id2idx[asin] for asin in st.session_state.cart if asin in id2idx]
                            
                            topk_idxs, topk_scores = recommend_top_k_beauty(
                                user_emb,
                                artifacts['checkpoint_data']['item_embs'],
                                k=UI_CONFIG['max_recommendations'],
                                exclude_idxs=history_idxs
                            )
                    
                    if user_emb is not None:
                        st.success("Recommendations ready")
                        
                        st.markdown("---")
                        st.markdown("### Your Personalized Recommendations")
                        
                        # Prepare data for table
                        recs_display = []
                        for rank, (idx, score) in enumerate(zip(topk_idxs, topk_scores), 1):
                            asin = artifacts["idx2item"][idx]
                            item_text = artifacts["items_df"].iloc[idx]["item_text"]
                            clean_text = clean_product_text(item_text)
                            
                            recs_display.append({
                                "Rank": rank,
                                "Product": clean_text,
                                "ASIN": asin,
                                "Match": f"{score:.1%}"
                            })
                        
                        # Display as table
                        df_display = pd.DataFrame(recs_display)
                        
                        st.dataframe(
                            df_display,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Rank": st.column_config.NumberColumn(
                                    "Rank",
                                    width="small",
                                    help="Recommendation rank"
                                ),
                                "Product": st.column_config.TextColumn(
                                    "Product",
                                    width="large",
                                    help="Product description"
                                ),
                                "ASIN": st.column_config.TextColumn(
                                    "ASIN",
                                    width="medium",
                                    help="Product code"
                                ),
                                "Match": st.column_config.TextColumn(
                                    "Match",
                                    width="small",
                                    help="Similarity score"
                                ),
                            },
                            height=400
                        )
                        
                        # Download
                        st.markdown("---")
                        recs_data = []
                        for rank, (idx, score) in enumerate(zip(topk_idxs, topk_scores), 1):
                            asin = artifacts["idx2item"][idx]
                            item_text = artifacts["items_df"].iloc[idx]["item_text"]
                            recs_data.append({
                                "Rank": rank,
                                "ASIN": asin,
                                "Score": f"{score:.4f}",
                                "Product": item_text
                            })
                        
                        csv = pd.DataFrame(recs_data).to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            f"recommendations_{selected_product.lower()}.csv",
                            "text/csv",
                            use_container_width=True
                        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #718096; padding: 1rem 0;'>"
        f"<p style='margin: 0; font-size: 0.9rem;'>Neural Retrieval • {selected_product} Recommendations</p>"
        f"</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()