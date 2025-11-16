# training/contrastive_loss.py
import torch
import torch.nn as nn

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, user_embs, item_embs):
        logits = user_embs @ item_embs.T / self.temp
        labels = torch.arange(len(user_embs), device=user_embs.device)
        return self.ce(logits, labels)
