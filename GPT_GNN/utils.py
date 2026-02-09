import ast
import csv
import os

import numpy as np
import scipy.sparse as sp
import torch
try:
    from texttable import Texttable
except Exception:  # optional dependency
    Texttable = None
from collections import OrderedDict

_USER_EMBED_CACHE = None
_USER_EMBED_DIM = None
_USER_EMBED_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "graph", "user_node_feature.csv")
)
_USER_EMBED_NPZ_PATH = _USER_EMBED_PATH.replace(".csv", ".npz")


def _load_user_embed_cache_csv(path):
    """原始 CSV 加载方式（慢，仅作备用）"""
    cache = {}
    embed_dim = None
    with open(path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            user = row[0]
            if len(row) < 2:
                continue
            vec_str = row[1]
            vec = np.fromstring(vec_str.strip("[]"), sep=",", dtype=np.float64)
            if vec.size == 0:
                vec = np.array(ast.literal_eval(vec_str), dtype=np.float64)
            if embed_dim is None:
                embed_dim = vec.size
            cache[user] = vec
    return cache, embed_dim


def _load_user_embed_cache_npz(path):
    """快速加载预处理的 NPZ 文件（推荐）"""
    data = np.load(path, allow_pickle=True)
    users = data['users']
    embeddings = data['embeddings']
    embed_dim = int(data['embed_dim'])
    
    # 构建字典缓存
    cache = {user: embeddings[i].astype(np.float64) for i, user in enumerate(users)}
    return cache, embed_dim


def _get_user_embed_cache():
    global _USER_EMBED_CACHE, _USER_EMBED_DIM
    if _USER_EMBED_CACHE is None:
        # 优先使用 NPZ 格式（快），否则回退到 CSV（慢）
        if os.path.exists(_USER_EMBED_NPZ_PATH):
            print(f"[INFO] 从 NPZ 加载用户嵌入缓存: {_USER_EMBED_NPZ_PATH}")
            _USER_EMBED_CACHE, _USER_EMBED_DIM = _load_user_embed_cache_npz(_USER_EMBED_NPZ_PATH)
        else:
            print(f"[WARN] NPZ 文件不存在，使用慢速 CSV 加载: {_USER_EMBED_PATH}")
            print(f"[WARN] 建议运行: python GPT_GNN/convert_embed_cache.py 生成 NPZ 文件")
            _USER_EMBED_CACHE, _USER_EMBED_DIM = _load_user_embed_cache_csv(_USER_EMBED_PATH)
    return _USER_EMBED_CACHE, _USER_EMBED_DIM

def args_print(args):
    _dict = vars(args)
    if Texttable is None:
        for k in _dict:
            print(f"{k}: {_dict[k]}")
        return
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        t.add_row([k, _dict[k]])
    print(t.draw())

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def randint():
    return np.random.randint(0, high=2 ** 32 - 1, dtype=np.int64)

# def feature_OAG(layer_data, graph):
#     feature = {}
#     times   = {}
#     indxs   = {}
#     texts   = []
#     for _type in layer_data:
#         if len(layer_data[_type]) == 0:
#             continue
#         idxs  = np.array(list(layer_data[_type].keys()))
#         tims  = np.array(list(layer_data[_type].values()))[:,1]
#
#         if 'node_emb' in graph.node_feature[_type]:
#             feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float64)
#         else:
#             feature[_type] = np.zeros([len(idxs), 400])
#         feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs, 'emb']),\
#             np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)
#
#         times[_type]   = tims
#         indxs[_type]   = idxs
#
#         if _type == 'paper':
#             attr = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
#     return feature, times, indxs, attr

def feature_OAG(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        # print(layer_data[_type])
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]
        # print(tims)   # 1
        # print(idxs)   # [idx, ]

        if _type == 'sub':
            # print(len(graph.node_feature['user'].loc[0, 'embed'])-len(graph.node_feature['sub'].loc[0, 'embed']))
            # 768-86=682
            feature[_type] = np.zeros([len(idxs), 682])
            feature[_type] = np.concatenate((list(graph.node_feature[_type].loc[idxs, 'embed']), feature[_type]), axis=1)
        else:
            name_list = list(graph.node_feature[_type].loc[idxs, 'id'])
            cache, embed_dim = _get_user_embed_cache()
            if embed_dim is None:
                raise ValueError("user_node_feature.csv is empty; cannot build user embeddings.")
            fallback = np.zeros(embed_dim, dtype=np.float64)
            embed_list = [cache.get(user, fallback) for user in name_list]
            feature[_type] = np.array(embed_list, dtype=np.float64)

        times[_type]   = tims
        indxs[_type]   = idxs

        if _type == 'sub':
            attr = feature[_type]
        # print(_type, len(feature[_type]))
        # print('embed', len(feature[_type][0]))     # 768, no problem
    return feature, times, indxs, attr


def feature_HIPT(layer_data, graph):
    feature = {}
    times = {}
    indxs = {}
    texts = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        # print(layer_data[_type])
        idxs = np.array(list(layer_data[_type].keys()))
        tims = np.array(list(layer_data[_type].values()))[:, 1]
        # print(tims)   # 1
        # print(idxs)   # [idx, ]

        if _type == 'sub':
            # print(len(graph.node_feature['user'].loc[0, 'embed'])-len(graph.node_feature['sub'].loc[0, 'embed']))
            # 768-86=682
            feature[_type] = np.zeros([len(idxs), 682])
            feature[_type] = np.concatenate((list(graph.node_feature[_type].loc[idxs, 'embed']), feature[_type]),
                                            axis=1)
        else:
            name_list = list(graph.node_feature[_type].loc[idxs, 'id'])
            cache, embed_dim = _get_user_embed_cache()
            if embed_dim is None:
                raise ValueError("user_node_feature.csv is empty; cannot build user embeddings.")
            fallback = np.zeros(embed_dim, dtype=np.float64)
            embed_list = [cache.get(user, fallback) for user in name_list]
            feature[_type] = np.array(embed_list, dtype=np.float64)

        times[_type] = tims
        indxs[_type] = idxs

        if _type == 'sub':
            attr = feature[_type]
        # print(_type, len(feature[_type]))
        # print('embed', len(feature[_type][0]))     # 768, no problem
    return feature, times, indxs, attr


def feature_DIY(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        # print(layer_data[_type])      # {tar_id:[ser, time]}
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]
        # print(tims)   # 1, 2, 11, 12
        # print(idxs)   # [idx, ]

        if _type == 'sub':
            # 86
            # feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'embed']), dtype=np.float64)

            # 768-86=682
            feature[_type] = np.zeros([len(idxs), 682])
            feature[_type] = np.concatenate((list(graph.node_feature[_type].loc[idxs, 'embed']), feature[_type]), axis=1)

        times[_type]   = tims
        indxs[_type]   = idxs

        if _type == 'sub':
            attr = feature[_type]
        # print(_type, len(feature[_type]))
        # print('embed', len(feature[_type][0]))     # 768, no problem
    return feature, times, indxs, attr


def feature_reddit(layer_data, graph):
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs  = np.array(list(layer_data[_type].keys()))
        tims  = np.array(list(layer_data[_type].values()))[:,1]

        feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'emb']), dtype=np.float64)
        times[_type]   = tims
        indxs[_type]   = idxs

        if _type == 'def':
            attr = feature[_type]
    return feature, times, indxs, attr

def load_gnn(_dict):
    out_dict = {}
    for key in _dict:
        if 'gnn' in key:
            out_dict[key[4:]] = _dict[key]
    return OrderedDict(out_dict)
