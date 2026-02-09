"""
Convert user_node_feature.csv into NumPy binary format.
Run once: python convert_embed_cache.py
"""
import os
import csv
import numpy as np
import ast
import time

# Path configuration.
CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "graph", "user_node_feature.csv")
)
NPZ_PATH = CSV_PATH.replace(".csv", ".npz")

def convert_csv_to_npz():
    print(f"Converting: {CSV_PATH}")
    print(f"Output file: {NPZ_PATH}")
    
    start_time = time.time()
    
    users = []
    embeddings = []
    embed_dim = None
    
    with open(CSV_PATH, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if not row or len(row) < 2:
                continue
            
            user = row[0]
            vec_str = row[1]
            
            # Parse embedding vector.
            vec = np.fromstring(vec_str.strip("[]"), sep=",", dtype=np.float32)
            if vec.size == 0:
                vec = np.array(ast.literal_eval(vec_str), dtype=np.float32)
            
            if embed_dim is None:
                embed_dim = vec.size
                print(f"Embedding dimension: {embed_dim}")
            
            users.append(user)
            embeddings.append(vec)
            
            if (i + 1) % 500000 == 0:
                print(f"  Processed {i + 1} rows...")
    
    print(f"Total users: {len(users)}")
    
    # Convert to NumPy arrays.
    users_array = np.array(users, dtype=object)
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    print(f"Embedding matrix shape: {embeddings_array.shape}")
    print(f"Memory usage: {embeddings_array.nbytes / 1024 / 1024:.1f} MB")
    
    # Save as NPZ.
    np.savez(NPZ_PATH, users=users_array, embeddings=embeddings_array, embed_dim=embed_dim)
    
    elapsed = time.time() - start_time
    print(f"Conversion completed! Elapsed: {elapsed:.1f}s")
    print(f"File size: {os.path.getsize(NPZ_PATH) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    convert_csv_to_npz()
