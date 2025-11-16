import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from utils.checkpoint import load_checkpoint
from training.precompute import precompute_item_embeddings
@torch.no_grad()
def evaluate_recall(model, seqs, item_texts, tokenizer, device, k_values=[5,10,20]):
    model.eval()
    data=load_checkpoint(model)
    if data:
        item_embs,item_ids,id2idx=data
    else:
        item_embs,item_ids,id2idx=precompute_item_embeddings(item_texts,tokenizer,model.item_model,device)
        item_embs=item_embs.to(device)
    recalls={k:0 for k in k_values}; mrr=0; n=0
    for seq in tqdm(seqs[:200],desc="Evaluating"):
        tgt=seq['target_item']
        if tgt not in id2idx: continue
        T=len(seq['sequence_items'])
        hist=torch.zeros(1,T,256,device=device)
        for t,a in enumerate(seq['sequence_items']):
            if a!='PAD' and a in id2idx: hist[0,t]=item_embs[id2idx[a]]
        q=torch.tensor([seq['quality_features']],dtype=torch.float32,device=device)
        m=torch.tensor([seq['sequence_mask']],dtype=torch.bool,device=device)
        u=model.user_model(hist,q,m)
        sims=torch.matmul(u,item_embs.T).squeeze(0)
        topk=torch.topk(sims,k=max(k_values)).indices.cpu().numpy()
        rank=np.where(topk==id2idx[tgt])[0]
        if len(rank)>0:
            r=rank[0]+1; mrr+=1.0/r
            for k in k_values:
                if r<=k: recalls[k]+=1
        n+=1
    print("Evaluation Results:")
    for k in k_values: print(f"Recall@{k}: {recalls[k]/n:.3f}")
    print(f"MRR: {mrr/n:.3f}")