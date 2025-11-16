import torch
import torch.nn as nn

class CombinedEmbeddingModel(nn.Module):
    def __init__(self, item_model, user_model):
        super().__init__()
        self.item_model = item_model
        self.user_model = user_model

    def forward(self, item_inputs, user_item_embs, q_feats, seq_mask):
        item_emb = self.item_model(**item_inputs)
        user_emb = self.user_model(user_item_embs, q_feats, seq_mask)
        return user_emb, item_emb