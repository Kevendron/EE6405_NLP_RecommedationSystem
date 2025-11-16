import torch
from tqdm import tqdm

@torch.no_grad()
def precompute_item_embeddings(item_texts, tokenizer, item_model, device, batch_size=64):
    item_ids = list(item_texts.keys())
    embs=[]
    item_model.eval()
    for i in tqdm(range(0,len(item_ids),batch_size),desc="Encoding items"):
        batch=item_ids[i:i+batch_size]
        toks=tokenizer([item_texts[x] for x in batch],
                       padding=True,truncation=True,max_length=128,return_tensors='pt')
        toks={k:v.to(device) for k,v in toks.items()}
        e=item_model(**toks)
        embs.append(e.cpu())
    embs=torch.cat(embs)
    id2idx={x:i for i,x in enumerate(item_ids)}
    return embs,item_ids,id2idx