"""
Beauty Product Two-Tower Recommender Model
Uses actual trained architecture with RoBERTa and Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaModel


class ItemEmbeddingModel(nn.Module):
    """
    Item tower - RoBERTa-based encoder
    Takes tokenized item text and outputs normalized embeddings
    """
    def __init__(self, model_name='roberta-base', embedding_dim=256):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.roberta.config.hidden_size, embedding_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        mean_emb = torch.sum(token_emb * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
        proj = self.projection(mean_emb)
        return F.normalize(proj, p=2, dim=1)


class UserEmbeddingModel(nn.Module):
    """
    User tower - Transformer encoder with quality features
    Takes sequence of item embeddings + quality features
    """
    def __init__(self, embedding_dim=256, num_layers=2, num_heads=8, max_seq_len=20):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.quality_encoder = nn.Sequential(
            nn.Linear(5, 32), 
            nn.ReLU(), 
            nn.Linear(32, 32)
        )
        d_model = embedding_dim + 32
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads,
            dim_feedforward=d_model*4, 
            dropout=0.1, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, d_model))

    def forward(self, item_embs, quality, mask):
        B, T, _ = item_embs.shape
        q_emb = self.quality_encoder(quality)
        x = torch.cat([item_embs, q_emb], dim=-1) + self.pos_enc[:, :T, :]
        pad_mask = ~mask.bool()
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (out * mask_f).sum(1) / torch.clamp(mask_f.sum(1), min=1e-9)
        return F.normalize(pooled[:, :self.embedding_dim], p=2, dim=1)


class CombinedEmbeddingModel(nn.Module):
    """Combined two-tower model"""
    def __init__(self, item_model, user_model):
        super().__init__()
        self.item_model = item_model
        self.user_model = user_model

    def forward(self, item_inputs, user_item_embs, q_feats, seq_mask):
        item_emb = self.item_model(**item_inputs)
        user_emb = self.user_model(user_item_embs, q_feats, seq_mask)
        return user_emb, item_emb


def load_beauty_checkpoint(checkpoint_path, device):
    """
    Load Beauty model checkpoint with full architecture
    Returns: dict with model, item_embs, item_ids, id2idx
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading Beauty checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create full model architecture
    item_model = ItemEmbeddingModel(model_name='roberta-base', embedding_dim=256)
    user_model = UserEmbeddingModel(embedding_dim=256, num_layers=2, num_heads=8, max_seq_len=20)
    model = CombinedEmbeddingModel(item_model, user_model)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.to(device)
    
    # Extract pre-computed embeddings and mappings
    item_embs = checkpoint['item_embs'].cpu().numpy()
    item_ids = checkpoint['item_ids']
    id2idx = checkpoint['id2idx']
    
    print(f"✓ Loaded {len(item_ids)} Beauty item embeddings")
    print(f"✓ Embedding dimension: {item_embs.shape[1]}")
    
    return {
        'model': model,
        'item_embs': item_embs,
        'item_ids': item_ids,
        'id2idx': id2idx,
        'device': device
    }


def get_user_embedding_beauty(history_asins, checkpoint_data):
    """
    Generate user embedding from interaction history
    For inference, we simplify by averaging item embeddings
    (In training, this goes through the full user tower with quality features)
    """
    item_embs = checkpoint_data['item_embs']
    id2idx = checkpoint_data['id2idx']
    
    history_idxs = []
    invalid_asins = []
    
    for asin in history_asins:
        if asin in id2idx:
            history_idxs.append(id2idx[asin])
        else:
            invalid_asins.append(asin)
    
    if len(history_idxs) == 0:
        return None, invalid_asins
    
    # Get historical item embeddings
    hist_embs = item_embs[history_idxs]
    
    # Simple averaging for user representation
    # (This is a simplified inference - training uses full Transformer with quality features)
    user_emb = np.mean(hist_embs, axis=0)
    user_emb = user_emb / np.linalg.norm(user_emb)  # L2 normalize
    
    return user_emb, invalid_asins


def recommend_top_k_beauty(user_emb, item_embs, k=10, exclude_idxs=None):
    """
    Get top-k recommendations using cosine similarity
    """
    sims = cosine_similarity(user_emb.reshape(1, -1), item_embs)[0]
    
    if exclude_idxs is not None:
        sims[exclude_idxs] = -1.0
    
    topk_idxs = np.argsort(sims)[::-1][:k]
    topk_scores = sims[topk_idxs]
    
    return topk_idxs, topk_scores