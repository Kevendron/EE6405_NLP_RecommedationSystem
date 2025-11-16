"""
Data Verification Script
Checks if all required files exist and have correct formats
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

def check_file_exists(path, description):
    """Check if file exists"""
    if path.exists():
        print(f"✅ {description}: Found at {path}")
        return True
    else:
        print(f"❌ {description}: NOT FOUND at {path}")
        return False

def verify_data_directory(data_dir):
    """Verify all required files and their formats"""
    
    data_path = Path(data_dir)
    artifacts_path = data_path / "optionB_nlp_model"
    
    print("=" * 60)
    print("DATA VERIFICATION REPORT")
    print("=" * 60)
    print(f"\nChecking directory: {data_path.absolute()}\n")
    
    all_good = True
    
    # 1. Check items.parquet
    print("1️⃣  Checking items.parquet...")
    items_path = data_path / "items.parquet"
    if check_file_exists(items_path, "items.parquet"):
        try:
            items_df = pd.read_parquet(items_path)
            print(f"   Shape: {items_df.shape}")
            print(f"   Columns: {list(items_df.columns)}")
            
            if "parent_asin" not in items_df.columns:
                print("   ⚠️  WARNING: 'parent_asin' column not found")
                all_good = False
            if "item_text" not in items_df.columns:
                print("   ⚠️  WARNING: 'item_text' column not found")
                all_good = False
            
            print(f"   Sample ASIN: {items_df.iloc[0]['parent_asin']}")
            print(f"   Sample text (first 80 chars): {items_df.iloc[0]['item_text'][:80]}...")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            all_good = False
    else:
        all_good = False
    
    print()
    
    # 2. Check mappings.json
    print("2️⃣  Checking mappings.json...")
    mappings_path = data_path / "mappings.json"
    if check_file_exists(mappings_path, "mappings.json"):
        try:
            with open(mappings_path, "r") as f:
                maps = json.load(f)
            
            if "item2idx" not in maps:
                print("   ❌ 'item2idx' key not found")
                all_good = False
            else:
                print(f"   Number of items: {len(maps['item2idx']):,}")
                print(f"   Sample mapping: {list(maps['item2idx'].items())[0]}")
            
            if "user2idx" in maps:
                print(f"   Number of users: {len(maps['user2idx']):,}")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            all_good = False
    else:
        all_good = False
    
    print()
    
    # 3. Check item_text_emb_384.npy
    print("3️⃣  Checking item_text_emb_384.npy...")
    base_emb_path = data_path / "item_text_emb_384.npy"
    if check_file_exists(base_emb_path, "item_text_emb_384.npy"):
        try:
            base_emb = np.load(base_emb_path)
            print(f"   Shape: {base_emb.shape}")
            print(f"   Dtype: {base_emb.dtype}")
            
            if base_emb.shape[1] != 384:
                print(f"   ⚠️  WARNING: Expected 384 dimensions, got {base_emb.shape[1]}")
                all_good = False
            
            print(f"   Sample values: {base_emb[0, :5]}")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
            all_good = False
    else:
        all_good = False
    
    print()
    
    # 4. Check artifacts directory
    print("4️⃣  Checking optionB_nlp_model/ directory...")
    if check_file_exists(artifacts_path, "optionB_nlp_model directory"):
        
        # 4a. Check item_projector.pt
        print("\n   4a. Checking item_projector.pt...")
        projector_path = artifacts_path / "item_projector.pt"
        if check_file_exists(projector_path, "   item_projector.pt"):
            try:
                state_dict = torch.load(projector_path, map_location="cpu")
                print(f"      Keys: {list(state_dict.keys())}")
                print(f"      First layer shape: {state_dict['mlp.0.weight'].shape}")
            except Exception as e:
                print(f"      ❌ Error loading file: {e}")
                all_good = False
        else:
            all_good = False
        
        print()
        
        # 4b. Check E_item_proj_128.npy
        print("   4b. Checking E_item_proj_128.npy...")
        proj_emb_path = artifacts_path / "E_item_proj_128.npy"
        if check_file_exists(proj_emb_path, "   E_item_proj_128.npy"):
            try:
                proj_emb = np.load(proj_emb_path)
                print(f"      Shape: {proj_emb.shape}")
                print(f"      Dtype: {proj_emb.dtype}")
                
                if proj_emb.shape[1] != 128:
                    print(f"      ⚠️  WARNING: Expected 128 dimensions, got {proj_emb.shape[1]}")
                    all_good = False
                
                print(f"      Sample values: {proj_emb[0, :5]}")
            except Exception as e:
                print(f"      ❌ Error reading file: {e}")
                all_good = False
        else:
            print("      ℹ️  This file can be auto-generated by the app")
    else:
        all_good = False
    
    print()
    
    # 5. Cross-validation checks
    print("5️⃣  Cross-validation checks...")
    try:
        items_df = pd.read_parquet(items_path)
        with open(mappings_path, "r") as f:
            maps = json.load(f)
        base_emb = np.load(base_emb_path)
        
        num_items_df = len(items_df)
        num_items_map = len(maps["item2idx"])
        num_items_emb = base_emb.shape[0]
        
        print(f"   Items in items.parquet: {num_items_df:,}")
        print(f"   Items in mappings.json: {num_items_map:,}")
        print(f"   Items in embeddings: {num_items_emb:,}")
        
        if num_items_df == num_items_map == num_items_emb:
            print("   ✅ All counts match!")
        else:
            print("   ❌ Mismatch in item counts!")
            all_good = False
            
    except Exception as e:
        print(f"   ⚠️  Could not perform cross-validation: {e}")
    
    print()
    print("=" * 60)
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready to run the app!")
        print("\nRun: streamlit run app.py")
    else:
        print("❌ SOME CHECKS FAILED - Please fix the issues above")
    print("=" * 60)
    
    return all_good

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "./data/Electronics"
    
    print(f"\nUsing data directory: {data_dir}")
    print("(Pass a different path as argument if needed)\n")
    
    verify_data_directory(data_dir)
