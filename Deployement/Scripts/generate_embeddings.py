"""
Helper Script: Generate Missing Embeddings
Run this if you don't have item_text_emb_384.npy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys

def generate_embeddings(data_dir, batch_size=128):
    """Generate MiniLM embeddings for all items"""
    
    data_path = Path(data_dir)
    
    print("=" * 60)
    print("GENERATING ITEM TEXT EMBEDDINGS")
    print("=" * 60)
    print(f"\nData directory: {data_path.absolute()}\n")
    
    # Load items
    items_path = data_path / "items.parquet"
    if not items_path.exists():
        print(f"❌ Error: items.parquet not found at {items_path}")
        return False
    
    print("Loading items.parquet...")
    items_df = pd.read_parquet(items_path)
    print(f"✅ Loaded {len(items_df):,} items")
    
    if "item_text" not in items_df.columns:
        print("❌ Error: 'item_text' column not found in items.parquet")
        return False
    
    # Check if embeddings already exist
    output_path = data_path / "item_text_emb_384.npy"
    if output_path.exists():
        response = input(f"\n⚠️  {output_path.name} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return False
    
    # Load MiniLM model
    print("\nLoading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded")
    
    # Extract texts
    texts = items_df['item_text'].tolist()
    print(f"\nGenerating embeddings for {len(texts):,} items...")
    print(f"Batch size: {batch_size}")
    
    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,  # We normalize later in the projector
        device='cuda' if model.device.type == 'cuda' else 'cpu'
    )
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    print(f"\n✅ Generated embeddings: {embeddings.shape}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Data type: {embeddings.dtype}")
    print(f"   Memory size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    # Save embeddings
    print(f"\nSaving to {output_path}...")
    np.save(output_path, embeddings)
    print("✅ Saved successfully!")
    
    # Verify
    print("\nVerifying saved file...")
    loaded = np.load(output_path)
    if np.allclose(loaded, embeddings):
        print("✅ Verification passed!")
    else:
        print("⚠️  Warning: Loaded embeddings differ from generated ones")
    
    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nFile saved to: {output_path}")
    print("You can now run: streamlit run app.py")
    
    return True

def generate_projected_embeddings(data_dir):
    """Generate projected embeddings if model is available"""
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    data_path = Path(data_dir)
    artifacts_path = data_path / "optionB_nlp_model"
    
    print("\n" + "=" * 60)
    print("GENERATING PROJECTED EMBEDDINGS")
    print("=" * 60)
    
    # Check if projector exists
    projector_path = artifacts_path / "item_projector.pt"
    if not projector_path.exists():
        print(f"⚠️  Projector not found at {projector_path}")
        print("Skipping projected embeddings generation.")
        return False
    
    # Load base embeddings
    base_emb_path = data_path / "item_text_emb_384.npy"
    if not base_emb_path.exists():
        print(f"❌ Base embeddings not found at {base_emb_path}")
        return False
    
    print(f"\nLoading base embeddings from {base_emb_path}...")
    base_emb = np.load(base_emb_path)
    print(f"✅ Loaded embeddings: {base_emb.shape}")
    
    # Define model architecture
    class ItemProjector(nn.Module):
        def __init__(self, in_dim=384, out_dim=128, hidden_dim=256):
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
    
    # Load projector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    projector = ItemProjector().to(device)
    projector.load_state_dict(torch.load(projector_path, map_location=device))
    projector.eval()
    print("✅ Loaded projector model")
    
    # Project embeddings
    print("\nProjecting embeddings...")
    base_emb_t = torch.from_numpy(base_emb).to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        proj_emb_t = projector(base_emb_t)
        proj_emb = proj_emb_t.cpu().numpy()
    
    print(f"✅ Projected embeddings: {proj_emb.shape}")
    
    # Save
    output_path = artifacts_path / "E_item_proj_128.npy"
    print(f"\nSaving to {output_path}...")
    np.save(output_path, proj_emb)
    print("✅ Saved successfully!")
    
    print("\n" + "=" * 60)
    print("PROJECTION COMPLETE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "./data/Electronics"
    
    print(f"\nData directory: {data_dir}")
    print("(Pass a different path as argument if needed)\n")
    
    # Generate base embeddings
    success = generate_embeddings(data_dir, batch_size=128)
    
    if success:
        # Try to generate projected embeddings
        print("\n")
        generate_projected_embeddings(data_dir)
