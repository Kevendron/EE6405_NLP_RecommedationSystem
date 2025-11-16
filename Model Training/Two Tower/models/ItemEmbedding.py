import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

# this module takes in the tokenized ids of the item metadata texts and outputs the normalized Item Embeddings 
class ItemEmbeddingModel(nn.Module):
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
    