from training.contrastive_loss import InfoNCELoss
import torch
import torch.nn as nn
from utils.checkpoint import save_checkpoint
from utils.checkpoint import load_checkpoint
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from training.precompute import precompute_item_embeddings

def train_model(model, seqs, item_texts, tokenizer, device, bs=8, epochs=2):
    item_ckpt = load_checkpoint(model)
    if item_ckpt:
        item_embs, item_ids, id2idx = item_ckpt
    else:
        item_embs, item_ids, id2idx = precompute_item_embeddings(item_texts, tokenizer, model.item_model, device)
        item_embs = item_embs.to(device)
        save_checkpoint(model, item_embs, item_ids, id2idx)

    loss_fn=InfoNCELoss()
    opt=optim.AdamW(model.parameters(),lr=1e-4)
    for ep in range(epochs):
        model.train(); total=steps=0
        for seq in tqdm(seqs[:500],desc=f"Epoch {ep+1}"):
            tgt=seq['target_item']
            if tgt not in id2idx: continue
            text=item_texts[tgt]
            toks=tokenizer(text,truncation=True,max_length=128,padding='max_length',return_tensors='pt')
            toks={k:v.to(device) for k,v in toks.items()}
            T=len(seq['sequence_items'])
            hist=torch.zeros(1,T,256,device=device)
            for t,a in enumerate(seq['sequence_items']):
                if a!='PAD' and a in id2idx: hist[0,t]=item_embs[id2idx[a]]
            q=torch.tensor([seq['quality_features']],dtype=torch.float32,device=device)
            m=torch.tensor([seq['sequence_mask']],dtype=torch.bool,device=device)
            u,i=model(toks,hist,q,m)
            loss=loss_fn(u,i)
            opt.zero_grad(); loss.backward(); opt.step()
            total+=loss.item(); steps+=1
        print(f"Epoch {ep+1}: avg loss={total/max(1,steps):.4f}")
        save_checkpoint(model, item_embs, item_ids, id2idx)
