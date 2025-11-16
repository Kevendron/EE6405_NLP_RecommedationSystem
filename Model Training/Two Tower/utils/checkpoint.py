import torch
from config import CHECKPOINT_DIR
import os
from utils.device import get_device
device = get_device()
def save_checkpoint(model, item_embs, item_ids, id2idx):
    os.makedirs(CHECKPOINT_DIR, exist_ok= True)
    torch.save({
        'model_state': model.state_dict(),
        'item_embs': item_embs.cpu(),
        'item_ids': item_ids,
        'id2idx': id2idx
    }, os.path.join(CHECKPOINT_DIR, "model_checkpoint.pt"))
    print("Model and embeddings saved to checkpoints/")


def load_checkpoint(model):
    path = os.path.join(CHECKPOINT_DIR, "model_checkpoint.pt")
    if not os.path.exists(path):
        print("No checkpoint found. Starting fresh.")
        return None
    data = torch.load(path, map_location=device)
    model.load_state_dict(data['model_state'])
    print("Loaded model checkpoint.")
    return data['item_embs'].to(device), data['item_ids'], data['id2idx']
