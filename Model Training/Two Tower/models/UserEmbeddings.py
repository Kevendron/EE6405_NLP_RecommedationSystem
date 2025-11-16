import torch
import torch.nn as nn
import torch.nn.functional as F


class UserEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=256, num_layers=2, num_heads=8, max_seq_len=20):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.quality_encoder = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32))
        d_model = embedding_dim + 32
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                               dim_feedforward=d_model*4, dropout=0.1, batch_first=True)
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