"""
将 user_node_feature.csv 转换为 numpy 二进制格式，大幅加速加载
运行一次即可：python convert_embed_cache.py
"""
import os
import csv
import numpy as np
import ast
import time

# 路径配置
CSV_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "graph", "user_node_feature.csv")
)
NPZ_PATH = CSV_PATH.replace(".csv", ".npz")

def convert_csv_to_npz():
    print(f"正在转换: {CSV_PATH}")
    print(f"输出文件: {NPZ_PATH}")
    
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
            
            # 解析向量
            vec = np.fromstring(vec_str.strip("[]"), sep=",", dtype=np.float32)
            if vec.size == 0:
                vec = np.array(ast.literal_eval(vec_str), dtype=np.float32)
            
            if embed_dim is None:
                embed_dim = vec.size
                print(f"嵌入维度: {embed_dim}")
            
            users.append(user)
            embeddings.append(vec)
            
            if (i + 1) % 500000 == 0:
                print(f"  已处理 {i + 1} 行...")
    
    print(f"总共 {len(users)} 个用户")
    
    # 转为 numpy 数组
    users_array = np.array(users, dtype=object)
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    print(f"嵌入矩阵形状: {embeddings_array.shape}")
    print(f"内存占用: {embeddings_array.nbytes / 1024 / 1024:.1f} MB")
    
    # 保存为 npz
    np.savez(NPZ_PATH, users=users_array, embeddings=embeddings_array, embed_dim=embed_dim)
    
    elapsed = time.time() - start_time
    print(f"转换完成! 耗时: {elapsed:.1f}s")
    print(f"文件大小: {os.path.getsize(NPZ_PATH) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    convert_csv_to_npz()
