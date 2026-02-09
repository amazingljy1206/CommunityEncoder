# -*- coding: utf-8 -*-
"""Unified fine-tune + evaluation pipeline for downstream tasks.

Notes (current built-ins):
  - Pipeline tasks: 3 (task1, task2, task3)
  - Pipeline methods: 4 total
      * task1: baseline
      * task2: baseline
      * task3: v2, v3
  - Model methods (GNN conv): use --conv_name, see --list.
  - Use --list for an always-accurate view (after adding new tasks/methods).

Usage examples:
  python inference_downstream/finetune_pipeline.py --list
  python inference_downstream/finetune_pipeline.py --task task1 --method baseline
"""

from __future__ import annotations

import argparse
import random
import sys
import os
# Add community_encoder to sys.path so scripts can import GPT_GNN from the parent directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from sklearn import metrics

from GPT_GNN.data import *
from GPT_GNN.model import *
from GPT_GNN.utils import _get_user_embed_cache

CONV_NAME_CHOICES = ["hgt", "gcn", "gat", "rgcn"]

# -----------------------------
# CLI helpers
# -----------------------------

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str):
    parser.add_argument(name, type=str2bool, nargs="?", const=True, default=default, help=help_text)

def _warm_user_embed_cache(enabled: bool):
    if not enabled:
        return
    _log("[CACHE] Warming user embedding cache...")
    _get_user_embed_cache()
    _log("[CACHE] User embedding cache ready.")

def _log_data_wait(epoch: int, n_epoch: int, n_batch: int, n_pool: int):
    _log(f"[DATA] Epoch {epoch}/{n_epoch}: collecting {n_batch} train batches + 1 valid (pool={n_pool})...")

def _collect_jobs(jobs, label: str, log_interval: int = 0):
    if log_interval <= 0:
        return [job.get() for job in jobs]
    start = time.time()
    results = []
    total = len(jobs)
    for idx, job in enumerate(jobs, 1):
        last_log = time.time()
        while not job.ready():
            if (time.time() - last_log) >= log_interval:
                elapsed = time.time() - start
                _log(f"[DATA] {label}: waiting {idx}/{total} ({elapsed:.1f}s elapsed)")
                last_log = time.time()
            time.sleep(1)
        results.append(job.get())
    return results

def _ensure_parent_dir(path: str):
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def _ensure_checkpoint(path: str, model, reason: str, logger: Optional["MetricsLogger"] = None):
    if not path or model is None:
        return
    if os.path.exists(path):
        return
    _log(f"[WARN] model_dir not found, saving current model to {path} ({reason})")
    _ensure_parent_dir(path)
    torch.save(model, path)
    if logger:
        logger.log_best(note=f"checkpoint_created:{reason}", path=path)

def _torch_load(path: str, weights_only: Optional[bool] = None, map_location: Optional[str] = None):
    if weights_only is None:
        if map_location is None:
            return torch.load(path)
        return torch.load(path, map_location=map_location)
    try:
        if map_location is None:
            return torch.load(path, weights_only=weights_only)
        return torch.load(path, weights_only=weights_only, map_location=map_location)
    except TypeError:
        # Older torch versions don't support weights_only
        if map_location is None:
            return torch.load(path)
        return torch.load(path, map_location=map_location)

def _log(message: str):
    print(message, flush=True)


class MetricsLogger:
    def __init__(self, path: str, args: argparse.Namespace):
        self.path = path
        _ensure_parent_dir(path)
        self.f = open(path, "a", encoding="utf-8")
        self._write(f"# run_start time={time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._write(f"# task={args.task} method={args.method}")
        self._write(f"# args={vars(args)}")

    def _write(self, line: str):
        self.f.write(line + "\n")
        self.f.flush()

    def log_epoch(self, **kwargs):
        self._write(_format_kv("epoch", **kwargs))

    def log_best(self, **kwargs):
        self._write(_format_kv("best", **kwargs))

    def log_test(self, **kwargs):
        self._write(_format_kv("test", **kwargs))

    def close(self):
        self._write(f"# run_end time={time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.f.close()


def _format_value(value):
    if isinstance(value, (float, np.floating)):
        return f"{value:.6f}"
    return str(value)


def _format_kv(prefix: str, **kwargs):
    parts = [prefix]
    for key, value in kwargs.items():
        parts.append(f"{key}={_format_value(value)}")
    return " ".join(parts)


def _resolve_log_path(args: argparse.Namespace):
    if getattr(args, "log_disable", False):
        return None
    if getattr(args, "log_file", None):
        return args.log_file
    if getattr(args, "model_dir", None):
        base = os.path.basename(args.model_dir)
        filename = base + ".log"
        model_parent = os.path.dirname(os.path.abspath(args.model_dir))
        log_dir = os.path.join(model_parent, "log")
        return os.path.join(log_dir, filename)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "log")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{args.task}_{args.method}_{stamp}.log"
    return os.path.join(log_dir, filename)


def _init_logger(args: argparse.Namespace):
    path = _resolve_log_path(args)
    if not path:
        return None
    return MetricsLogger(path, args)


# -----------------------------
# Registry
# -----------------------------

@dataclass
class MethodSpec:
    name: str
    description: str
    add_args: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], None]
    aliases: Tuple[str, ...] = ()


@dataclass
class TaskSpec:
    name: str
    description: str
    default_method: str
    add_args: Callable[[argparse.ArgumentParser], None]
    methods: Dict[str, MethodSpec] = field(default_factory=dict)
    aliases: Tuple[str, ...] = ()

    def add_method(
        self,
        name: str,
        description: str,
        add_args: Callable[[argparse.ArgumentParser], None],
        run: Callable[[argparse.Namespace], None],
        aliases: Iterable[str] = (),
    ) -> MethodSpec:
        method = MethodSpec(
            name=name,
            description=description,
            add_args=add_args,
            run=run,
            aliases=tuple(aliases),
        )
        self.methods[name] = method
        for alias in method.aliases:
            METHOD_ALIASES[(self.name, alias)] = name
        return method


TASKS: Dict[str, TaskSpec] = {}
TASK_ALIASES: Dict[str, str] = {}
METHOD_ALIASES: Dict[Tuple[str, str], str] = {}


def register_task(
    name: str,
    description: str,
    default_method: str,
    add_args: Callable[[argparse.ArgumentParser], None],
    aliases: Iterable[str] = (),
) -> TaskSpec:
    task = TaskSpec(
        name=name,
        description=description,
        default_method=default_method,
        add_args=add_args,
        aliases=tuple(aliases),
    )
    TASKS[name] = task
    for alias in task.aliases:
        TASK_ALIASES[alias] = name
    return task


# -----------------------------
# Shared state for multiprocessing
# -----------------------------

_TASK1_STATE = {}
_TASK2_STATE = {}
_TASK3_STATE = {}
_TASK3_V3_PRE_GNN = None
_TASK3_V3_PRE_GNN_SIG = None
_TASK3_V3_PRE_STATE = None
_TASK3_V3_THREADS_SET = False
_SOC_DIM_STATE = {}
_COM_CONF_STATE = {}


def _resolve_device(cuda_id: int) -> torch.device:
    if cuda_id is None or cuda_id == -1:
        return torch.device("cpu")
    return torch.device(f"cuda:{cuda_id}")


def _set_numpy_seed(seed: Optional[int]):
    if seed is None:
        return
    np.random.seed(seed)


def _pad_node_feature(node_feature: torch.Tensor, target_dim: int) -> torch.Tensor:
    if node_feature.size(1) >= target_dim:
        return node_feature
    pad = torch.zeros(
        node_feature.size(0),
        target_dim - node_feature.size(1),
        dtype=node_feature.dtype,
        device=node_feature.device,
    )
    return torch.cat([node_feature, pad], dim=1)


def _infer_embed_dim(graph, node_type: str) -> int:
    if node_type not in graph.node_feature or len(graph.node_feature[node_type]) == 0:
        raise ValueError(f"Missing node features for type '{node_type}'")
    sample = graph.node_feature[node_type]["embed"].values[0]
    return len(sample)


def _balanced_sample_nodes(nodes, labels, batch_size: int, pos_ratio: float):
    nodes = np.array(nodes)
    labels = np.array(labels)
    if len(nodes) == 0:
        return nodes

    pos_nodes = nodes[labels == 1]
    neg_nodes = nodes[labels == 0]

    if len(pos_nodes) == 0 or len(neg_nodes) == 0:
        replace = len(nodes) < batch_size
        return np.random.choice(nodes, batch_size, replace=replace)

    pos_count = int(round(batch_size * pos_ratio))
    pos_count = max(0, min(batch_size, pos_count))
    neg_count = batch_size - pos_count

    pos_sample = np.random.choice(pos_nodes, pos_count, replace=len(pos_nodes) < pos_count)
    neg_sample = np.random.choice(neg_nodes, neg_count, replace=len(neg_nodes) < neg_count)
    sample_nodes = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(sample_nodes)
    return sample_nodes


def _feature_from_graph(layer_data, graph):
    feature = {}
    times = {}
    indxs = {}
    texts = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs = np.array(list(layer_data[_type].keys()))
        tims = np.array(list(layer_data[_type].values()))[:, 1]
        feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, "embed"]), dtype=np.float64)
        times[_type] = tims
        indxs[_type] = idxs
    return feature, times, indxs, None


# -----------------------------
# Task 1: user -> sub multilabel
# -----------------------------

def task1_node_classification_sample(seed, pairs, time_range):
    args = _TASK1_STATE["args"]
    graph = _TASK1_STATE["graph"]
    cand_list = _TASK1_STATE["cand_list"]
    target_type = _TASK1_STATE["target_type"]

    np.random.seed(seed)
    target_ids = np.random.choice(list(pairs.keys()), args.batch_size, replace=False)
    target_info = []
    for target_id in target_ids:
        _, _time = pairs[target_id]
        target_info += [[target_id, _time]]

    feature, times, edge_list, _, _ = sample_subgraph(
        graph,
        time_range,
        inp={target_type: np.array(target_info)},
        sampled_depth=args.sample_depth,
        sampled_number=args.sample_width,
    )

    masked_edge_list = []
    for i in edge_list["user"]["sub"]["sub_user"]:
        if i[0] >= args.batch_size:
            masked_edge_list += [i]
    edge_list["user"]["sub"]["sub_user"] = masked_edge_list

    masked_edge_list = []
    for i in edge_list["sub"]["user"]["sub_user"]:
        if i[1] >= args.batch_size:
            masked_edge_list += [i]
    edge_list["sub"]["user"]["sub_user"] = masked_edge_list

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = to_torch(
        feature, times, edge_list, graph
    )

    ylabel = np.zeros([args.batch_size, len(cand_list)])
    for x_id, target_id in enumerate(target_ids):
        if target_id not in pairs:
            print("error 1" + str(target_id))
        for source_id in pairs[target_id][0]:
            if source_id not in cand_list:
                print("error 2" + str(target_id))
            ylabel[x_id][cand_list.index(source_id)] = 1

    my_label = ylabel
    ylabel /= ylabel.sum(axis=1).reshape(-1, 1)
    x_ids = np.arange(args.batch_size) + node_dict["user"][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, my_label


def run_task1_baseline(args: argparse.Namespace):
    _set_numpy_seed(args.seed)
    device = _resolve_device(args.cuda)
    logger = _init_logger(args)
    model = None

    graph = renamed_load(open(args.data_dir, "rb"))

    target_type = "user"
    cand_list = list(graph.edge_list["sub"]["user"]["sub_user"].keys())
    _TASK1_STATE["args"] = args
    _TASK1_STATE["graph"] = graph
    _TASK1_STATE["cand_list"] = cand_list
    _TASK1_STATE["target_type"] = target_type

    pre_range = {t: True for t in graph.times if t is not None and t == 0}
    train_range = {t: True for t in graph.times if t is not None and t == 1}
    valid_range = {t: True for t in graph.times if t is not None and t == 2}
    test_range = {t: True for t in graph.times if t is not None and t == 3}

    train_pairs = {}
    valid_pairs = {}
    test_pairs = {}
    for target_id in graph.edge_list["user"]["sub"]["sub_user"]:
        for source_id in graph.edge_list["user"]["sub"]["sub_user"][target_id]:
            _time = graph.edge_list["user"]["sub"]["sub_user"][target_id][source_id]
            if _time in train_range:
                if target_id not in train_pairs:
                    train_pairs[target_id] = [[], _time]
                train_pairs[target_id][0] += [source_id]
            elif _time in valid_range:
                if target_id not in valid_pairs:
                    valid_pairs[target_id] = [[], _time]
                valid_pairs[target_id][0] += [source_id]
            else:
                if target_id not in test_pairs:
                    test_pairs[target_id] = [[], _time]
                test_pairs[target_id][0] += [source_id]

    sel_train_pairs = {
        p: train_pairs[p]
        for p in np.random.choice(
            list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage), replace=False
        )
    }
    sel_valid_pairs = {
        p: valid_pairs[p]
        for p in np.random.choice(
            list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage), replace=False
        )
    }

    criterion = nn.KLDivLoss(reduction="batchmean")

    def prepare_data(pool):
        jobs = []
        for _ in np.arange(args.n_batch):
            jobs.append(pool.apply_async(task1_node_classification_sample, args=(randint(), sel_train_pairs, train_range)))
        jobs.append(pool.apply_async(task1_node_classification_sample, args=(randint(), sel_valid_pairs, valid_range)))
        return jobs

    test_only = bool(getattr(args, "test_only", False) or getattr(args, "test_mode", False))

    if not test_only:
        _warm_user_embed_cache(args.warm_cache)
        pool = mp.Pool(args.n_pool)
        st = time.time()
        jobs = prepare_data(pool)

        gnn = GNN(
            conv_name=args.conv_name,
            in_dim=len(graph.node_feature["user"]["embed"].values[0]),
            n_hid=args.n_hid,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            num_types=len(graph.get_types()),
            num_relations=len(graph.get_meta_graph()) + 1,
            prev_norm=args.prev_norm,
            last_norm=args.last_norm,
        )

        if args.use_pretrain:
            gnn.load_state_dict(load_gnn(torch.load(args.pretrain_model_dir)), strict=False)
            _log("Load Pre-trained Model from (%s)" % args.pretrain_model_dir)
        else:
            _log("No pretrain model loaded")

        classifier = Classifier(args.n_hid, len(cand_list))
        if args.keep_train:
            print("Keep-training Model from (%s)" % args.keep_train_dir)
            model = _torch_load(args.keep_train_dir, weights_only=False)
            gnn, classifier = model
        else:
            model = nn.Sequential(gnn, classifier).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

        best_val = 0
        best_epoch = 0
        train_step = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)

        for epoch in np.arange(args.n_epoch) + 1:
            _log_data_wait(int(epoch), args.n_epoch, args.n_batch, args.n_pool)
            st = time.time()
            train_data = _collect_jobs(jobs[:-1], "train", args.data_wait_log_sec)
            valid_data = _collect_jobs([jobs[-1]], "valid", args.data_wait_log_sec)[0]
            et = time.time()
            # _log("Data Preparation: %.1fs" % (et - st))
            jobs = prepare_data(pool)

            model.train()
            train_losses = []
            for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, my_label in train_data:
                node_rep = gnn.forward(
                    node_feature.to(device),
                    node_type.to(device),
                    edge_time.to(device),
                    edge_index.to(device),
                    edge_type.to(device),
                )
                res = classifier.forward(node_rep[x_ids])
                loss = criterion(res, torch.FloatTensor(ylabel).to(device))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

                train_losses += [loss.cpu().detach().tolist()]
                train_step += 1
                scheduler.step()
                del res, loss

            model.eval()
            with torch.no_grad():
                node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, my_label = valid_data
                node_rep = gnn.forward(
                    node_feature.to(device),
                    node_type.to(device),
                    edge_time.to(device),
                    edge_index.to(device),
                    edge_type.to(device),
                )
                res = classifier.forward(node_rep[x_ids])
                loss = criterion(res, torch.FloatTensor(ylabel).to(device))

                valid_res = []
                for ai, bi in zip(ylabel, res.argsort(descending=True)):
                    valid_res += [ai[bi.cpu().numpy()]]
                valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])

                st = time.time()
                _log(
                    (
                        "Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  "
                        "Valid Loss: %.2f  Valid NDCG: %.4f"
                    )
                    % (
                        epoch,
                        (st - et),
                        optimizer.param_groups[0]["lr"],
                        np.average(train_losses),
                        loss.cpu().detach().tolist(),
                        valid_ndcg,
                    )
                )
                if logger:
                    logger.log_epoch(
                        epoch=int(epoch),
                        lr=optimizer.param_groups[0]["lr"],
                        train_loss=float(np.average(train_losses)),
                        valid_loss=float(loss.cpu().detach().tolist()),
                        valid_ndcg=float(valid_ndcg),
                    )

                if valid_ndcg > best_val:
                    best_val = valid_ndcg
                    best_epoch = int(epoch)
                    _ensure_parent_dir(args.model_dir)
                    torch.save(model, args.model_dir)
                    _log("[[[UPDATE!!!]]]")
                    if logger:
                        logger.log_best(
                            epoch=best_epoch,
                            best_valid_ndcg=float(best_val),
                        )

                del res, loss
            del train_data, valid_data
        pool.close()
        pool.join()
    else:
        _log("test mode")

    if not test_only:
        _ensure_checkpoint(args.model_dir, model, "task1_final", logger)
    best_model = _torch_load(args.model_dir, weights_only=False)
    best_model.eval()
    gnn, classifier = best_model
    with torch.no_grad():
        test_res = []
        for _ in range(10):
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, my_label = task1_node_classification_sample(
                randint(), test_pairs, test_range
            )
            paper_rep = gnn.forward(
                node_feature.to(device),
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )[x_ids]
            res = classifier.forward(paper_rep)
            for ai, bi in zip(ylabel, res.argsort(descending=True)):
                test_res += [ai[bi.cpu().numpy()]]
        test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
        _log("Best Test NDCG: %.4f" % np.average(test_ndcg))
        test_mrr = mean_reciprocal_rank(test_res)
        _log("Best Test MRR:  %.4f" % np.average(test_mrr))
        if logger:
            logger.log_test(
                test_ndcg=float(np.average(test_ndcg)),
                test_mrr=float(np.average(test_mrr)),
                best_valid_ndcg=float(best_val),
                best_epoch=int(best_epoch),
            )
            logger.close()



# -----------------------------
# Task 2: sub-sub edge sentiment
# -----------------------------

def _task2_build_edge_labels(edge_list, indx, graph):
    labels = []
    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                for ti, si in edge_list[target_type][source_type][relation_type]:
                    if relation_type == "self":
                        labels.append(0)
                        continue
                    tid = indx[target_type][ti]
                    sid = indx[source_type][si]
                    try:
                        labels.append(graph.edge_list[target_type][source_type][relation_type][tid][sid])
                    except KeyError:
                        labels.append(0)
    return labels

def task2_node_classification_sample(seed, pairs, time_range):
    args = _TASK2_STATE["args"]
    graph = _TASK2_STATE["graph"]
    target_type = _TASK2_STATE["target_type"]

    np.random.seed(seed)
    target_ids = np.random.choice(list(pairs.keys()), args.batch_size, replace=False)
    target_info = []
    for target_id in target_ids:
        source_ids, _time = pairs[target_id]
        target_info += [[target_id, _time]]
        _ = np.random.choice(source_ids, 1, replace=False)[0]

    feature, times, edge_list, indx, texts = sample_subgraph_DIY(
        graph,
        time_range,
        inp={target_type: np.array(target_info)},
        sampled_depth=args.sample_depth,
        sampled_number=args.sample_width,
        feature_extractor=feature_DIY,
    )

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = to_torch_DIY(
        feature, times, edge_list, graph
    )
    edge_label = _task2_build_edge_labels(edge_list, indx, graph)

    t_s_label = []
    s_t_label = []
    for index, pair in enumerate(edge_index.t()):
        sid, tid = int(pair[0]), int(pair[1])
        label = edge_label[index]
        if sid < args.batch_size and tid < args.batch_size and sid != tid:
            t_s_label.append([tid, sid, label])
            s_t_label.append([sid, tid, label])

    ylabel = np.zeros([len(t_s_label), 1])
    for x_id, t_s_list in enumerate(t_s_label):
        tid, sid, label = t_s_list[0], t_s_list[1], t_s_list[2]
        if indx["sub"][tid] not in target_ids or indx["sub"][sid] not in target_ids:
            print("error")
        if label != 0 and label % 2 == 0:
            ylabel[x_id][0] = 1

    x_ids = np.arange(len(t_s_label)) + node_dict["sub"][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, t_s_label


def task2_upsample(node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, t_s_label):
    args = _TASK2_STATE["args"]
    total_label = len(t_s_label)
    total_pos = np.sum(ylabel)
    sample_num = int(total_label - 2 * total_pos)
    to_sample = []
    for idx in x_ids:
        if t_s_label[idx][2] % 2 == 0:
            to_sample.append(idx)
    sample_idx = np.random.choice(to_sample, sample_num, replace=True)

    x_ids = np.arange(total_label + sample_num)
    ylabel = np.concatenate((ylabel, ylabel[sample_idx]), axis=0)
    t_s_label = np.array(t_s_label)
    t_s_label = np.concatenate((t_s_label, t_s_label[sample_idx])).tolist()

    shuffle = list(zip(ylabel, t_s_label))
    random.shuffle(shuffle)
    ylabel[:], t_s_label[:] = zip(*shuffle)

    if len(ylabel) > args.batch_size:
        ylabel = ylabel[: args.batch_size]
        t_s_label = t_s_label[: args.batch_size]
    ylabel = np.array(ylabel)
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, t_s_label


def task2_concatenation(node_rep, t_s_label):
    cat_list = []
    for t_s_list in t_s_label:
        tid, sid, _label = t_s_list[0], t_s_list[1], t_s_list[2]
        cat = torch.cat([node_rep[tid], node_rep[sid]], dim=0)
        cat_list.append(cat)
    concatenation_rep = torch.stack(cat_list, dim=0)
    return concatenation_rep


def run_task2_baseline(args: argparse.Namespace):
    _set_numpy_seed(args.seed)
    device = _resolve_device(args.cuda)
    logger = _init_logger(args)
    model = None

    graph = renamed_load(open(args.data_dir, "rb"))

    target_type = "sub"
    _TASK2_STATE["args"] = args
    _TASK2_STATE["graph"] = graph
    _TASK2_STATE["target_type"] = target_type

    pre_range = {t: True for t in graph.times if t is not None and (t == 1 or t == 2)}
    train_range = {t: True for t in graph.times if t is not None and (t == 11 or t == 12)}
    valid_range = {t: True for t in graph.times if t is not None and (t == 21 or t == 22)}
    test_range = {t: True for t in graph.times if t is not None and (t == 31 or t == 32)}

    train_pairs = {}
    valid_pairs = {}
    test_pairs = {}
    for target_id in graph.edge_list["sub"]["sub"]["sub_sub"]:
        for source_id in graph.edge_list["sub"]["sub"]["sub_sub"][target_id]:
            _time = graph.edge_list["sub"]["sub"]["sub_sub"][target_id][source_id]
            if _time in train_range:
                if target_id not in train_pairs:
                    train_pairs[target_id] = [[], _time]
                train_pairs[target_id][0] += [source_id]
            elif _time in valid_range:
                if target_id not in valid_pairs:
                    valid_pairs[target_id] = [[], _time]
                valid_pairs[target_id][0] += [source_id]
            else:
                if target_id not in test_pairs:
                    test_pairs[target_id] = [[], _time]
                test_pairs[target_id][0] += [source_id]

    sel_train_pairs = {
        p: train_pairs[p]
        for p in np.random.choice(
            list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage), replace=False
        )
    }
    sel_valid_pairs = {
        p: valid_pairs[p]
        for p in np.random.choice(
            list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage), replace=False
        )
    }

    criterion = nn.CrossEntropyLoss()

    def prepare_data(pool):
        jobs = []
        for _ in np.arange(args.n_batch):
            jobs.append(pool.apply_async(task2_node_classification_sample, args=(randint(), sel_train_pairs, train_range)))
        jobs.append(pool.apply_async(task2_node_classification_sample, args=(randint(), sel_valid_pairs, valid_range)))
        return jobs

    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)

    gnn = GNN(
        conv_name=args.conv_name,
        in_dim=768,
        n_hid=args.n_hid,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        num_types=2,
        num_relations=5,
        prev_norm=args.prev_norm,
        last_norm=args.last_norm,
    )

    if args.use_pretrain:
        gnn.load_state_dict(load_gnn(torch.load(args.pretrain_model_dir)), strict=False)
        _log("Load Pre-trained Model from (%s)" % args.pretrain_model_dir)
    else:
        _log("No pretrain model loaded")

    classifier = ClassifierDIY(args.n_hid * 2, 1)
    if args.keep_train:
        print("Keep-training Model from (%s)" % args.keep_train_dir)
        model = _torch_load(args.keep_train_dir, weights_only=False)
        gnn, classifier = model
    else:
        model = nn.Sequential(gnn, classifier).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    best_val = 0
    best_f1 = 0
    best_acc = 0
    best_record = None
    train_step = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)

    for epoch in np.arange(args.n_epoch) + 1:
        if args.test_only:
            _log("test mode")
            break

        _log_data_wait(int(epoch), args.n_epoch, args.n_batch, args.n_pool)
        st = time.time()
        train_data = _collect_jobs(jobs[:-1], "train", args.data_wait_log_sec)
        valid_data = _collect_jobs([jobs[-1]], "valid", args.data_wait_log_sec)[0]
        et = time.time()
        # _log("Data Preparation: %.1fs" % (et - st))
        jobs = prepare_data(pool)

        model.train()
        train_losses = []
        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, t_s_label in train_data:
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, t_s_label = task2_upsample(
                node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, t_s_label
            )

            node_rep = gnn.forward(
                node_feature.to(device),
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )
            concate_rep = task2_concatenation(node_rep, t_s_label)
            res = classifier.forward(concate_rep)

            _ylabel = torch.FloatTensor(ylabel).squeeze(dim=1)
            loss = criterion(res, _ylabel.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step()
            del res, loss

        model.eval()
        with torch.no_grad():
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, t_s_label = valid_data
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, t_s_label = task2_upsample(
                node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, t_s_label
            )

            node_rep = gnn.forward(
                node_feature.to(device),
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )
            concate_rep = task2_concatenation(node_rep, t_s_label)
            res = classifier.forward(concate_rep)
            _ylabel = torch.FloatTensor(ylabel).squeeze(dim=1)
            loss = criterion(res, _ylabel.to(device))

            true = ylabel.reshape(1, -1)[0].tolist()
            pre = np.around(res.cpu()).tolist()
            val_pos_rate = float(np.mean(true)) if len(true) else 0.0
            pred_pos_rate = float(np.mean(pre)) if len(pre) else 0.0
            acc = metrics.accuracy_score(true, pre)
            valid_f1 = metrics.f1_score(true, pre, pos_label=1)
            fpr, tpr, thresholds = metrics.roc_curve(true, pre, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            st = time.time()
            _log(
                (
                    "Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  "
                    "Valid Acc: %.4f  Valid F1: %.4f Valid AUC: %.4f"
                )
                % (
                    epoch,
                    (st - et),
                    optimizer.param_groups[0]["lr"],
                    np.average(train_losses),
                    loss.cpu().detach().tolist(),
                    acc,
                    valid_f1,
                    auc,
                )
            )
            if logger:
                logger.log_epoch(
                    epoch=int(epoch),
                    lr=optimizer.param_groups[0]["lr"],
                    train_loss=float(np.average(train_losses)),
                    valid_loss=float(loss.cpu().detach().tolist()),
                    valid_acc=float(acc),
                    valid_f1=float(valid_f1),
                    valid_auc=float(auc),
                    credit=args.credit,
                    val_pos_rate=val_pos_rate,
                    pred_pos_rate=pred_pos_rate,
                )

            if args.credit == "auc":
                credit = auc
            elif args.credit == "acc":
                credit = acc
            else:
                credit = valid_f1

            if credit == best_val:
                if valid_f1 > best_f1:
                    best_f1 = valid_f1
                    best_record = {
                        "epoch": int(epoch),
                        "best_credit": float(credit),
                        "valid_acc": float(acc),
                        "valid_f1": float(valid_f1),
                        "valid_auc": float(auc),
                        "credit_metric": args.credit,
                    }
                    _ensure_parent_dir(args.model_dir)
                    torch.save(model, args.model_dir)
                    _log("[[[UPDATE!!!]]]")
                    if logger:
                        logger.log_best(**best_record)
                if valid_f1 == best_f1:
                    if acc > best_acc:
                        best_acc = acc
                        best_record = {
                            "epoch": int(epoch),
                            "best_credit": float(credit),
                            "valid_acc": float(acc),
                            "valid_f1": float(valid_f1),
                            "valid_auc": float(auc),
                            "credit_metric": args.credit,
                        }
                        _ensure_parent_dir(args.model_dir)
                        torch.save(model, args.model_dir)
                        _log("[[[UPDATE!!!]]]")
                        if logger:
                            logger.log_best(**best_record)
            if credit > best_val:
                best_val = credit
                best_record = {
                    "epoch": int(epoch),
                    "best_credit": float(credit),
                    "valid_acc": float(acc),
                    "valid_f1": float(valid_f1),
                    "valid_auc": float(auc),
                    "credit_metric": args.credit,
                }
                _ensure_parent_dir(args.model_dir)
                torch.save(model, args.model_dir)
                _log("[[[UPDATE!!!]]]")
                if logger:
                    logger.log_best(**best_record)

            del res, loss
        del train_data, valid_data
    pool.close()
    pool.join()

    if not args.test_only:
        _ensure_checkpoint(args.model_dir, model, "task2_final", logger)
    best_model = _torch_load(args.model_dir, weights_only=False)
    best_model.eval()
    gnn, classifier = best_model
    with torch.no_grad():
        test_acc = []
        test_f1 = []
        test_auc = []
        for _ in range(20):
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel, t_s_label = task2_node_classification_sample(
                randint(), test_pairs, test_range
            )

            paper_rep = gnn.forward(
                node_feature.to(device),
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )

            concate_rep = task2_concatenation(paper_rep, t_s_label)
            res = classifier.forward(concate_rep)

            true = ylabel.reshape(1, -1)[0].tolist()
            pre = np.around(res.cpu()).tolist()
            total_pos = np.sum(true)
            total_label = len(t_s_label)
            _ = total_pos / total_label * 1.0
            if total_pos == 0:
                continue

            acc = metrics.accuracy_score(true, pre)
            valid_f1 = metrics.f1_score(true, pre, pos_label=1)
            fpr, tpr, thresholds = metrics.roc_curve(true, pre, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            test_acc.append(acc)
            test_f1.append(valid_f1)
            test_auc.append(auc)

        _log("Best Test Acc: %.4f" % np.average(test_acc))
        _log("Best Test F1:  %.4f" % np.average(test_f1))
        _log("Best Test AUC:  %.4f" % np.average(test_auc))
        _log(str(len(test_auc)))
        if logger:
            logger.log_test(
                test_acc=float(np.average(test_acc)),
                test_f1=float(np.average(test_f1)),
                test_auc=float(np.average(test_auc)),
                best_credit=float(best_val),
            )
            logger.close()


# -----------------------------
# Task 3: sub node binary classification
# -----------------------------

def task3_node_classification_sample_v2(seed, nodes, time_range):
    args = _TASK3_STATE["args"]
    graph = _TASK3_STATE["graph"]

    np.random.seed(seed)
    pos_idx = np.nonzero(graph.y[nodes])[0]
    pos_nodes = nodes[pos_idx]
    neg_nodes = np.delete(nodes, pos_idx)
    pos_samp_nodes = np.random.choice(pos_nodes, int(args.batch_size * 0.25), replace=False)
    neg_samp_nodes = np.random.choice(neg_nodes, int(args.batch_size * 0.75), replace=False)

    samp_nodes = np.append(pos_samp_nodes, neg_samp_nodes)
    np.random.shuffle(samp_nodes)

    feature, times, edge_list, _, texts = sample_subgraph(
        graph,
        time_range,
        inp={
            "sub": np.concatenate([samp_nodes, np.ones(args.batch_size)]).reshape(2, -1).transpose()
        },
        sampled_depth=args.sample_depth,
        sampled_number=args.sample_width,
        feature_extractor=feature_OAG,
    )

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = to_torch(
        feature, times, edge_list, graph
    )

    x_ids = np.arange(args.batch_size)
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, graph.y[samp_nodes]


def task3_node_classification_sample_v3(seed, nodes, time_range):
    args = _TASK3_STATE["args"]
    graph = _TASK3_STATE["graph"]
    global _TASK3_V3_THREADS_SET
    if getattr(args, "torch_threads", None) is not None and not _TASK3_V3_THREADS_SET:
        try:
            torch.set_num_threads(int(args.torch_threads))
            torch.set_num_interop_threads(int(args.torch_threads))
        except Exception:
            pass
        _TASK3_V3_THREADS_SET = True

    np.random.seed(seed)
    pos_idx = np.nonzero(graph.y[nodes])[0]
    pos_nodes = nodes[pos_idx]
    neg_nodes = np.delete(nodes, pos_idx)
    pos_samp_nodes = np.random.choice(pos_nodes, int(args.batch_size * 0.25), replace=False)
    neg_samp_nodes = np.random.choice(neg_nodes, int(args.batch_size * 0.75), replace=False)

    samp_nodes = np.append(pos_samp_nodes, neg_samp_nodes)

    feature, times, edge_list, _, texts = sample_subgraph(
        graph,
        time_range,
        inp={
            "sub": np.concatenate([samp_nodes, np.ones(args.batch_size)]).reshape(2, -1).transpose()
        },
        sampled_depth=args.sample_depth,
        sampled_number=args.sample_width,
        feature_extractor=feature_OAG,
    )

    dim_sub = feature["sub"].shape[0]
    dim_user = feature["user"].shape[0]

    if args.use_pretrain:
        global _TASK3_V3_PRE_GNN, _TASK3_V3_PRE_GNN_SIG
        sig = (
            args.pretrain_model_dir,
            args.conv_name,
            args.n_hid,
            args.n_heads,
            args.n_layers,
            args.dropout,
            args.prev_norm,
            args.last_norm,
        )
        if _TASK3_V3_PRE_GNN is None or _TASK3_V3_PRE_GNN_SIG != sig:
            pre_gnn = GNN(
                conv_name=args.conv_name,
                in_dim=768,
                n_hid=args.n_hid,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                dropout=args.dropout,
                num_types=2,
                num_relations=5,
                prev_norm=args.prev_norm,
                last_norm=args.last_norm,
            )
            pre_state = _TASK3_V3_PRE_STATE
            if pre_state is None:
                pre_state = _torch_load(args.pretrain_model_dir, weights_only=False, map_location="cpu")
            pre_gnn.load_state_dict(load_gnn(pre_state), strict=False)
            pre_gnn.eval()
            _TASK3_V3_PRE_GNN = pre_gnn
            _TASK3_V3_PRE_GNN_SIG = sig
        pre_gnn = _TASK3_V3_PRE_GNN

        node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = to_torch(
            feature, times, edge_list, graph
        )
        with torch.no_grad():
            pre_rep = pre_gnn.forward(node_feature, node_type, edge_time, edge_index, edge_type)
        pre_rep = pre_rep.detach().numpy()

        idx_sub = np.arange(dim_sub)
        feature["sub"] = np.concatenate((feature["sub"], pre_rep[idx_sub]), axis=1)

        idx_user = np.arange(dim_sub, dim_user + dim_sub)
        feature["user"] = np.concatenate((feature["user"], pre_rep[idx_user]), axis=1)
    else:
        blank_feature = np.zeros((dim_sub, 400))
        feature["sub"] = np.concatenate((feature["sub"], blank_feature), axis=1)

        blank_feature1 = np.zeros((dim_user, 400))
        feature["user"] = np.concatenate((feature["user"], blank_feature1), axis=1)

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = to_torch(
        feature, times, edge_list, graph
    )

    x_ids = np.arange(args.batch_size)
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, graph.y[samp_nodes]


def run_task3_common(args: argparse.Namespace, sample_fn, in_dim: int, load_pretrain_to_gnn: bool):
    device = _resolve_device(args.cuda)
    logger = _init_logger(args)
    graph = renamed_load(open(args.data_dir, "rb"))
    model = None

    # Preload v3 pretrain weights once in main process to avoid repeated disk IO in workers.
    if args.method == "v3" and args.use_pretrain:
        global _TASK3_V3_PRE_STATE
        if _TASK3_V3_PRE_STATE is None:
            _log("[CACHE] Preloading v3 pretrain weights on CPU...")
            _TASK3_V3_PRE_STATE = _torch_load(args.pretrain_model_dir, weights_only=False, map_location="cpu")
            _log("[CACHE] v3 pretrain weights ready.")

    _TASK3_STATE["args"] = args
    _TASK3_STATE["graph"] = graph

    train_target_nodes = graph.train_target_nodes
    valid_target_nodes = graph.valid_target_nodes
    test_target_nodes = graph.test_target_nodes

    criterion = nn.BCELoss()

    _warm_user_embed_cache(args.warm_cache)

    def prepare_data(pool):
        jobs = []
        for _ in np.arange(args.n_batch):
            jobs.append(pool.apply_async(sample_fn, args=(randint(), train_target_nodes, {1: True})))
        jobs.append(pool.apply_async(sample_fn, args=(randint(), valid_target_nodes, {1: True})))
        return jobs

    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)

    gnn = GNN(
        conv_name=args.conv_name,
        in_dim=in_dim,
        n_hid=args.n_hid,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        num_types=2,
        num_relations=5,
        prev_norm=args.prev_norm,
        last_norm=args.last_norm,
    )

    if args.use_pretrain:
        if load_pretrain_to_gnn:
            gnn.load_state_dict(load_gnn(torch.load(args.pretrain_model_dir)), strict=False)
        _log("Load Pre-trained Model from (%s)" % args.pretrain_model_dir)
    else:
        _log("No pretrain model loaded")

    classifier = ClassifierDIY(args.n_hid, 1)
    if args.keep_train:
        print("Keep-training Model from (%s)" % args.keep_train_dir)
        model = _torch_load(args.keep_train_dir, weights_only=False)
        gnn, classifier = model
    else:
        model = nn.Sequential(gnn, classifier).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    best_val = 0.0001 if load_pretrain_to_gnn else -100
    best_record = None
    train_step = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)

    for epoch in np.arange(args.n_epoch) + 1:
        if args.test_only:
            _log("test mode")
            break

        _log_data_wait(int(epoch), args.n_epoch, args.n_batch, args.n_pool)
        st = time.time()
        train_data = _collect_jobs(jobs[:-1], "train", args.data_wait_log_sec)
        valid_data = _collect_jobs([jobs[-1]], "valid", args.data_wait_log_sec)[0]
        et = time.time()
        # _log("Data Preparation: %.1fs" % (et - st))
        jobs = prepare_data(pool)

        model.train()
        train_losses = []
        torch.cuda.empty_cache()
        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
            node_rep = gnn.forward(
                node_feature.to(device),
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )
            res = classifier.forward(node_rep[x_ids])

            _ylabel = torch.Tensor(ylabel)
            loss = criterion(res, _ylabel.to(device))
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step()
            del res, loss

        model.eval()
        with torch.no_grad():
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
            node_rep = gnn.forward(
                node_feature.to(device),
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )

            res = classifier.forward(node_rep[x_ids])
            _ylabel = torch.Tensor(ylabel)
            loss = criterion(res, _ylabel.to(device))

            true = ylabel.tolist()
            pre = np.around(res.cpu()).tolist()
            val_pos_rate = float(np.mean(true)) if len(true) else 0.0
            pred_pos_rate = float(np.mean(pre)) if len(pre) else 0.0
            valid_f1 = metrics.f1_score(true, pre, pos_label=1)
            acc = metrics.accuracy_score(true, pre)
            fpr, tpr, thresholds = metrics.roc_curve(true, pre, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            st = time.time()
            _log(
                (
                    "Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  "
                    "Valid Acc: %.4f  Valid F1: %.4f Valid AUC: %.4f"
                )
                % (
                    epoch,
                    (st - et),
                    optimizer.param_groups[0]["lr"],
                    np.average(train_losses),
                    loss.cpu().detach().tolist(),
                    acc,
                    valid_f1,
                    auc,
                )
            )
            if logger:
                logger.log_epoch(
                    epoch=int(epoch),
                    lr=optimizer.param_groups[0]["lr"],
                    train_loss=float(np.average(train_losses)),
                    valid_loss=float(loss.cpu().detach().tolist()),
                    valid_acc=float(acc),
                    valid_f1=float(valid_f1),
                    valid_auc=float(auc),
                    credit=args.credit,
                    val_pos_rate=val_pos_rate,
                    pred_pos_rate=pred_pos_rate,
                )

            if args.credit == "auc":
                credit = auc
            elif args.credit == "acc":
                credit = acc
            else:
                credit = valid_f1

            if credit > best_val:
                best_val = credit
                best_record = {
                    "epoch": int(epoch),
                    "best_credit": float(credit),
                    "valid_acc": float(acc),
                    "valid_f1": float(valid_f1),
                    "valid_auc": float(auc),
                    "credit_metric": args.credit,
                }
                _ensure_parent_dir(args.model_dir)
                torch.save(model, args.model_dir)
                _log("[[[UPDATE!!!]]]")
                if logger:
                    logger.log_best(**best_record)

            del res, loss
        del train_data, valid_data
    pool.close()
    pool.join()

    if not args.test_only:
        _ensure_checkpoint(args.model_dir, model, "task3_final", logger)
    best_model = _torch_load(args.model_dir, weights_only=False)
    best_model.eval()
    gnn, classifier = best_model
    with torch.no_grad():
        test_acc = []
        test_f1 = []
        test_auc = []
        test_p = []
        test_r = []
        for _ in range(20):
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = sample_fn(
                randint(), test_target_nodes, {1: True}
            )

            paper_rep = gnn.forward(
                node_feature.to(device),
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )[x_ids]

            res = classifier.forward(paper_rep)

            true = ylabel.reshape(1, -1)[0].tolist()
            pre = np.around(res.cpu()).tolist()

            acc = metrics.accuracy_score(true, pre)
            valid_f1 = metrics.f1_score(true, pre, pos_label=1)
            fpr, tpr, thresholds = metrics.roc_curve(true, pre, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            recall = metrics.recall_score(true, pre)
            precision = metrics.precision_score(true, pre)

            test_acc.append(acc)
            test_f1.append(valid_f1)
            test_auc.append(auc)
            test_r.append(recall)
            test_p.append(precision)

        _log("Best Test Acc: %.4f" % np.average(test_acc))
        _log("Best Test F1:  %.4f" % np.average(test_f1))
        _log("Best Test AUC:  %.4f" % np.average(test_auc))
        if load_pretrain_to_gnn:
            _log("Best Test recall:  %.4f" % np.average(test_r))
            _log("Best Test precision:  %.4f" % np.average(test_p))
        if logger:
            logger.log_test(
                test_acc=float(np.average(test_acc)),
                test_f1=float(np.average(test_f1)),
                test_auc=float(np.average(test_auc)),
                best_credit=float(best_val),
            )
            logger.close()


# -----------------------------
# Task soc_dim / com_conf: binary node classification
# -----------------------------

def _binary_node_classification_sample(seed, nodes, time_range, state):
    args = state["args"]
    graph = state["graph"]
    target_type = state["target_type"]

    np.random.seed(seed)
    nodes = np.array(nodes)
    labels = graph.y[nodes]
    samp_nodes = _balanced_sample_nodes(nodes, labels, args.batch_size, args.pos_ratio)
    if len(samp_nodes) != args.batch_size:
        replace = len(nodes) < args.batch_size
        samp_nodes = np.random.choice(nodes, args.batch_size, replace=replace)

    target_info = np.stack([samp_nodes, np.ones(len(samp_nodes), dtype=np.int64)], axis=1)

    feature, times, edge_list, _, _ = sample_subgraph(
        graph,
        time_range,
        inp={target_type: target_info},
        sampled_depth=args.sample_depth,
        sampled_number=args.sample_width,
        feature_extractor=_feature_from_graph,
    )

    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = to_torch(
        feature, times, edge_list, graph
    )
    x_ids = np.arange(args.batch_size) + node_dict[target_type][0]
    return node_feature, node_type, edge_time, edge_index, edge_type, x_ids, graph.y[samp_nodes]


def soc_dim_node_classification_sample(seed, nodes, time_range):
    return _binary_node_classification_sample(seed, nodes, time_range, _SOC_DIM_STATE)


def com_conf_node_classification_sample(seed, nodes, time_range):
    return _binary_node_classification_sample(seed, nodes, time_range, _COM_CONF_STATE)


def _resolve_binary_task_splits(graph, seed: Optional[int]):
    if all(hasattr(graph, name) for name in ("train_target_nodes", "valid_target_nodes", "test_target_nodes")):
        return graph.train_target_nodes, graph.valid_target_nodes, graph.test_target_nodes

    idx = np.arange(len(graph.y))
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(idx)
    train = idx[: int(len(idx) * 0.8)]
    valid = idx[int(len(idx) * 0.8) : int(len(idx) * 0.9)]
    test = idx[int(len(idx) * 0.9) :]
    return train, valid, test


def _resolve_gnn_dims(graph, args: argparse.Namespace, base_in_dim: int):
    num_types = len(graph.get_types())
    num_relations = len(graph.get_meta_graph()) + 1
    use_pretrain = bool(getattr(args, "use_pretrain", False))

    if use_pretrain and args.conv_name != "hgt":
        _log("[WARN] use_pretrain only supported with HGT; disabling pretrain.")
        use_pretrain = False

    in_dim = base_in_dim
    if use_pretrain:
        pre_dim = getattr(args, "pretrain_in_dim", base_in_dim)
        if base_in_dim > pre_dim:
            _log("[WARN] base feature dim > pretrain_in_dim; disabling pretrain.")
            return base_in_dim, num_types, num_relations, False
        in_dim = max(base_in_dim, pre_dim)
        pre_types = getattr(args, "pretrain_num_types", None)
        pre_rels = getattr(args, "pretrain_num_relations", None)
        if pre_types is not None:
            num_types = max(num_types, int(pre_types))
        if pre_rels is not None:
            num_relations = max(num_relations, int(pre_rels))

    return in_dim, num_types, num_relations, use_pretrain


def _run_binary_node_task(args: argparse.Namespace, state: dict, target_type: str, sample_fn):
    _set_numpy_seed(args.seed)
    device = _resolve_device(args.cuda)
    logger = _init_logger(args)
    graph = renamed_load(open(args.data_dir, "rb"))
    state["args"] = args
    state["graph"] = graph
    state["target_type"] = target_type

    train_nodes, valid_nodes, test_nodes = _resolve_binary_task_splits(graph, args.seed)
    if getattr(args, "data_percentage", 1.0) < 1.0:
        train_nodes = np.random.choice(
            train_nodes, int(len(train_nodes) * args.data_percentage), replace=False
        )
        valid_nodes = np.random.choice(
            valid_nodes, int(len(valid_nodes) * args.data_percentage), replace=False
        )

    base_in_dim = _infer_embed_dim(graph, target_type)
    in_dim, num_types, num_relations, use_pretrain = _resolve_gnn_dims(graph, args, base_in_dim)

    criterion = nn.BCELoss()

    def prepare_data(pool):
        jobs = []
        for _ in np.arange(args.n_batch):
            jobs.append(pool.apply_async(sample_fn, args=(randint(), train_nodes, {1: True})))
        jobs.append(pool.apply_async(sample_fn, args=(randint(), valid_nodes, {1: True})))
        return jobs

    pool = mp.Pool(args.n_pool)
    jobs = prepare_data(pool)

    gnn = GNN(
        conv_name=args.conv_name,
        in_dim=in_dim,
        n_hid=args.n_hid,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        num_types=num_types,
        num_relations=num_relations,
        prev_norm=args.prev_norm,
        last_norm=args.last_norm,
    )

    if use_pretrain:
        try:
            gnn.load_state_dict(load_gnn(_torch_load(args.pretrain_model_dir, weights_only=False)), strict=False)
            _log("Load Pre-trained Model from (%s)" % args.pretrain_model_dir)
        except Exception as exc:
            _log(f"[WARN] Failed to load pretrain weights: {exc}")
            use_pretrain = False
    if not use_pretrain:
        _log("No pretrain model loaded")

    classifier = ClassifierDIY(args.n_hid, 1)
    if args.keep_train:
        print("Keep-training Model from (%s)" % args.keep_train_dir)
        model = _torch_load(args.keep_train_dir, weights_only=False)
        gnn, classifier = model
    else:
        model = nn.Sequential(gnn, classifier).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    best_val = -100.0
    best_record = None
    train_step = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=1e-6)

    for epoch in np.arange(args.n_epoch) + 1:
        if args.test_only:
            _log("test mode")
            break

        _log_data_wait(int(epoch), args.n_epoch, args.n_batch, args.n_pool)
        st = time.time()
        train_data = _collect_jobs(jobs[:-1], "train", args.data_wait_log_sec)
        valid_data = _collect_jobs([jobs[-1]], "valid", args.data_wait_log_sec)[0]
        et = time.time()
        jobs = prepare_data(pool)

        model.train()
        train_losses = []
        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
            node_feature = node_feature.to(device)
            if node_feature.size(1) < in_dim:
                node_feature = _pad_node_feature(node_feature, in_dim)

            node_rep = gnn.forward(
                node_feature,
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )
            res = classifier.forward(node_rep[x_ids])

            _ylabel = torch.Tensor(ylabel).to(device)
            loss = criterion(res, _ylabel)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step()
            del res, loss

        model.eval()
        with torch.no_grad():
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
            node_feature = node_feature.to(device)
            if node_feature.size(1) < in_dim:
                node_feature = _pad_node_feature(node_feature, in_dim)

            node_rep = gnn.forward(
                node_feature,
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )
            res = classifier.forward(node_rep[x_ids])
            _ylabel = torch.Tensor(ylabel).to(device)
            loss = criterion(res, _ylabel)

            true = ylabel.tolist()
            pre = np.around(res.cpu()).tolist()
            val_pos_rate = float(np.mean(true)) if len(true) else 0.0
            pred_pos_rate = float(np.mean(pre)) if len(pre) else 0.0
            valid_f1 = metrics.f1_score(true, pre, pos_label=1)
            acc = metrics.accuracy_score(true, pre)
            fpr, tpr, thresholds = metrics.roc_curve(true, pre, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            st = time.time()
            _log(
                (
                    "Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  "
                    "Valid Acc: %.4f  Valid F1: %.4f Valid AUC: %.4f"
                )
                % (
                    epoch,
                    (st - et),
                    optimizer.param_groups[0]["lr"],
                    np.average(train_losses),
                    loss.cpu().detach().tolist(),
                    acc,
                    valid_f1,
                    auc,
                )
            )
            if logger:
                logger.log_epoch(
                    epoch=int(epoch),
                    lr=optimizer.param_groups[0]["lr"],
                    train_loss=float(np.average(train_losses)),
                    valid_loss=float(loss.cpu().detach().tolist()),
                    valid_acc=float(acc),
                    valid_f1=float(valid_f1),
                    valid_auc=float(auc),
                    credit=args.credit,
                    val_pos_rate=val_pos_rate,
                    pred_pos_rate=pred_pos_rate,
                )

            if args.credit == "auc":
                credit = auc
            elif args.credit == "acc":
                credit = acc
            else:
                credit = valid_f1

            if credit > best_val:
                best_val = credit
                best_record = {
                    "epoch": int(epoch),
                    "best_credit": float(credit),
                    "valid_acc": float(acc),
                    "valid_f1": float(valid_f1),
                    "valid_auc": float(auc),
                    "credit_metric": args.credit,
                }
                _ensure_parent_dir(args.model_dir)
                torch.save(model, args.model_dir)
                _log("[[[UPDATE!!!]]]")
                if logger:
                    logger.log_best(**best_record)

            del res, loss
        del train_data, valid_data
    pool.close()
    pool.join()

    if not args.test_only:
        _ensure_checkpoint(args.model_dir, model, "binary_task_final", logger)
    best_model = _torch_load(args.model_dir, weights_only=False)
    best_model.eval()
    gnn, classifier = best_model
    with torch.no_grad():
        test_acc = []
        test_f1 = []
        test_auc = []
        for _ in range(20):
            node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = sample_fn(
                randint(), test_nodes, {1: True}
            )

            node_feature = node_feature.to(device)
            if node_feature.size(1) < in_dim:
                node_feature = _pad_node_feature(node_feature, in_dim)

            paper_rep = gnn.forward(
                node_feature,
                node_type.to(device),
                edge_time.to(device),
                edge_index.to(device),
                edge_type.to(device),
            )[x_ids]

            res = classifier.forward(paper_rep)

            true = ylabel.reshape(1, -1)[0].tolist()
            pre = np.around(res.cpu()).tolist()

            acc = metrics.accuracy_score(true, pre)
            valid_f1 = metrics.f1_score(true, pre, pos_label=1)
            fpr, tpr, thresholds = metrics.roc_curve(true, pre, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            test_acc.append(acc)
            test_f1.append(valid_f1)
            test_auc.append(auc)

        _log("Best Test Acc: %.4f" % np.average(test_acc))
        _log("Best Test F1:  %.4f" % np.average(test_f1))
        _log("Best Test AUC:  %.4f" % np.average(test_auc))
        if logger:
            logger.log_test(
                test_acc=float(np.average(test_acc)),
                test_f1=float(np.average(test_f1)),
                test_auc=float(np.average(test_auc)),
                best_credit=float(best_val),
            )
            logger.close()


def run_soc_dim_baseline(args: argparse.Namespace):
    _run_binary_node_task(args, _SOC_DIM_STATE, target_type="sub", sample_fn=soc_dim_node_classification_sample)


def run_com_conf_baseline(args: argparse.Namespace):
    _run_binary_node_task(args, _COM_CONF_STATE, target_type="user", sample_fn=com_conf_node_classification_sample)


# -----------------------------
# Argument builders
# -----------------------------


def add_soc_dim_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../graph/soc_dim_graph_bert.pk",
        choices=["../graph/soc_dim_graph_bert.pk", "../graph/soc_dim_graph.pk"],
    )
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--sample_depth", type=int, default=2)
    parser.add_argument("--sample_width", type=int, default=32)

    add_bool_arg(parser, "--use_pretrain", default=True, help_text="Whether to use pre-trained model (HGT only)")
    parser.add_argument("--pretrain_model_dir", type=str, default="../models/model3_epoch10l_stage2.pt")
    parser.add_argument("--pretrain_in_dim", type=int, default=768)
    parser.add_argument("--pretrain_num_types", type=int, default=2)
    parser.add_argument("--pretrain_num_relations", type=int, default=5)
    add_bool_arg(parser, "--keep_train", default=False, help_text="Resume training from checkpoint")
    parser.add_argument("--keep_train_dir", type=str, default="../models/finetune_models/**.pth")
    parser.add_argument("--model_dir", type=str, default="../models/finetune_models/soc_dim/epoch50_hipt_stage2.pth")
    add_bool_arg(parser, "--test_only", default=False, help_text="Run evaluation only")
    parser.add_argument("--credit", type=str, default="auc", choices=["acc", "auc", "f1"])
    parser.add_argument("--pos_ratio", type=float, default=0.5)

    parser.add_argument("--conv_name", type=str, default="hgt", choices=CONV_NAME_CHOICES)
    parser.add_argument("--n_hid", type=int, default=400)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--prev_norm", action="store_true")
    parser.add_argument("--last_norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd", "adagrad"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cycle", "cosine"])
    parser.add_argument("--data_percentage", type=float, default=1.0)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--n_pool", type=int, default=6)
    parser.add_argument("--n_batch", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=43)


def add_com_conf_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../graph/com_conf_graph.pk",
        choices=["../graph/com_conf_graph.pk", "../sample/com_conf_graph.pk"],
    )
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--sample_depth", type=int, default=2)
    parser.add_argument("--sample_width", type=int, default=32)

    add_bool_arg(parser, "--use_pretrain", default=True, help_text="Whether to use pre-trained model (HGT only)")
    parser.add_argument("--pretrain_model_dir", type=str, default="../models/model3_epoch10l_stage2.pt")
    parser.add_argument("--pretrain_in_dim", type=int, default=768)
    parser.add_argument("--pretrain_num_types", type=int, default=2)
    parser.add_argument("--pretrain_num_relations", type=int, default=5)
    add_bool_arg(parser, "--keep_train", default=False, help_text="Resume training from checkpoint")
    parser.add_argument("--keep_train_dir", type=str, default="../models/finetune_models/**.pth")
    parser.add_argument("--model_dir", type=str, default="../models/finetune_models/com_conf_hgt.pth")
    add_bool_arg(parser, "--test_only", default=False, help_text="Run evaluation only")
    parser.add_argument("--credit", type=str, default="f1", choices=["acc", "auc", "f1"])
    parser.add_argument("--pos_ratio", type=float, default=0.5)

    parser.add_argument("--conv_name", type=str, default="hgt", choices=CONV_NAME_CHOICES)
    parser.add_argument("--n_hid", type=int, default=400)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--prev_norm", action="store_true")
    parser.add_argument("--last_norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd", "adagrad"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cycle", "cosine"])
    parser.add_argument("--data_percentage", type=float, default=1.0)
    parser.add_argument("--n_epoch", type=int, default=30)
    parser.add_argument("--n_pool", type=int, default=6)
    parser.add_argument("--n_batch", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=43)


def add_task1_args(parser: argparse.ArgumentParser):
    parser.add_argument("--data_dir", type=str, default="../sample/graph_sample_time.pk")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--sample_depth", type=int, default=6)
    parser.add_argument("--sample_width", type=int, default=64)

    add_bool_arg(parser, "--use_pretrain", default=True, help_text="Whether to use pre-trained model")
    parser.add_argument("--pretrain_model_dir", type=str, default="../models/model3_epoch10l_current.pt")
    add_bool_arg(parser, "--keep_train", default=False, help_text="Resume training from checkpoint")
    parser.add_argument("--keep_train_dir", type=str, default="../models/finetune_models/**.pth")
    parser.add_argument("--model_dir", type=str, default="../models/rebuttal_model/sub_only_link_1.pth")
    add_bool_arg(parser, "--test_mode", default=False, help_text="Run evaluation only")
    add_bool_arg(parser, "--test_only", default=False, help_text="Alias for --test_mode")

    parser.add_argument("--conv_name", type=str, default="hgt", choices=CONV_NAME_CHOICES)
    parser.add_argument("--n_hid", type=int, default=400)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--prev_norm", action="store_true")
    parser.add_argument("--last_norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd", "adagrad"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cycle", "cosine"])
    parser.add_argument("--data_percentage", type=float, default=1.0)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--n_pool", type=int, default=6)
    parser.add_argument("--n_batch", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=43)


def add_task2_args(parser: argparse.ArgumentParser):
    parser.add_argument("--data_dir", type=str, default="../graph/graph_sub_label2.pk")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--sample_depth", type=int, default=6)
    parser.add_argument("--sample_width", type=int, default=64)

    add_bool_arg(parser, "--use_pretrain", default=True, help_text="Whether to use pre-trained model")
    parser.add_argument("--pretrain_model_dir", type=str, default="../models/model2_sample60u2.pt")
    add_bool_arg(parser, "--keep_train", default=False, help_text="Resume training from checkpoint")
    parser.add_argument("--keep_train_dir", type=str, default="../models/finetune_models/task1_sample_pre_base50.pth")
    parser.add_argument("--model_dir", type=str, default="../models/finetune_models/task2_sample_stage1.pth")
    add_bool_arg(parser, "--test_only", default=False, help_text="Run evaluation only")
    parser.add_argument("--credit", type=str, default="acc", choices=["acc", "auc", "f1"])

    parser.add_argument("--conv_name", type=str, default="hgt", choices=CONV_NAME_CHOICES)
    parser.add_argument("--n_hid", type=int, default=400)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--prev_norm", action="store_true")
    parser.add_argument("--last_norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd", "adagrad"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cycle", "cosine"])
    parser.add_argument("--data_percentage", type=float, default=1.0)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--n_pool", type=int, default=6)
    parser.add_argument("--n_batch", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=43)


def add_task3_v2_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../graph/graph_sub_label2-2.pk",
        choices=["../graph/graph_sub_label2-2.pk", "../sample/graph_sample2-2.pk", "../graph/graph_sub_label2-2_extra.pk"],
    )
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--sample_depth", type=int, default=4)
    parser.add_argument("--sample_width", type=int, default=64)

    add_bool_arg(parser, "--use_pretrain", default=True, help_text="Whether to use pre-trained model")
    parser.add_argument("--pretrain_model_dir", type=str, default="../models/rebuttal_model/user_only_node.pt")
    add_bool_arg(parser, "--keep_train", default=False, help_text="Resume training from checkpoint")
    parser.add_argument("--keep_train_dir", type=str, default="../models/finetune_models/**.pth")
    parser.add_argument("--model_dir", type=str, default="../models/rebuttal_model/user_only_node_2.pth")
    add_bool_arg(parser, "--test_only", default=False, help_text="Run evaluation only")
    parser.add_argument("--credit", type=str, default="f1", choices=["acc", "auc", "f1"])

    parser.add_argument("--conv_name", type=str, default="hgt", choices=CONV_NAME_CHOICES)
    parser.add_argument("--n_hid", type=int, default=400)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--prev_norm", action="store_true")
    parser.add_argument("--last_norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd", "adagrad"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cycle", "cosine"])
    parser.add_argument("--data_percentage", type=float, default=1.0)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--n_pool", type=int, default=6)
    parser.add_argument("--n_batch", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip", type=float, default=0.2)


def add_task3_v3_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../sample/graph_sample2-2.pk",
        choices=["../graph/graph_sub_label2-2.pk", "../sample/graph_sample2-2.pk"],
    )
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--sample_depth", type=int, default=6)
    parser.add_argument("--sample_width", type=int, default=64)

    add_bool_arg(parser, "--use_pretrain", default=True, help_text="Whether to use pre-trained model")
    parser.add_argument("--pretrain_model_dir", type=str, default="../models/model3_epoch10_stage2.pt")
    add_bool_arg(parser, "--keep_train", default=False, help_text="Resume training from checkpoint")
    parser.add_argument("--keep_train_dir", type=str, default="../models/finetune_models/**.pth")
    parser.add_argument("--model_dir", type=str, default="../models/finetune_models/task2-3_sample_model3_10-2.pth")
    add_bool_arg(parser, "--test_only", default=False, help_text="Run evaluation only")
    parser.add_argument("--credit", type=str, default="f1", choices=["acc", "auc", "f1"])

    parser.add_argument("--conv_name", type=str, default="hgt", choices=CONV_NAME_CHOICES)
    parser.add_argument("--n_hid", type=int, default=400)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--prev_norm", action="store_true")
    parser.add_argument("--last_norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd", "adagrad"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cycle", "cosine"])
    parser.add_argument("--data_percentage", type=float, default=1.0)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--n_pool", type=int, default=6)
    parser.add_argument("--n_batch", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--clip", type=float, default=0.5)


# -----------------------------
# Task registration
# -----------------------------

TASK1 = register_task(
    name="task1",
    description="User -> sub multilabel classification (sample1)",
    default_method="baseline",
    add_args=add_task1_args,
    aliases=("sample1", "task1_user_sub"),
)
TASK1.add_method(
    name="baseline",
    description="Original sample1 pipeline",
    add_args=lambda p: None,
    run=run_task1_baseline,
    aliases=("default",),
)

TASK2 = register_task(
    name="task2",
    description="Sub-sub edge sentiment classification (sample2)",
    default_method="baseline",
    add_args=add_task2_args,
    aliases=("sample2", "task2_sub_sub"),
)
TASK2.add_method(
    name="baseline",
    description="Original sample2 pipeline",
    add_args=lambda p: None,
    run=run_task2_baseline,
    aliases=("default",),
)

TASK3 = register_task(
    name="task3",
    description="Sub node binary classification (sample2-2/2-3)",
    default_method="v2",
    add_args=lambda p: None,
    aliases=("sample2-2", "sample2-3", "task3_sub_node"),
)
TASK3.add_method(
    name="v2",
    description="Version 2-2 pipeline",
    add_args=add_task3_v2_args,
    run=lambda args: run_task3_common(args, task3_node_classification_sample_v2, in_dim=768, load_pretrain_to_gnn=True),
    aliases=("2-2", "sample2-2"),
)
TASK3.add_method(
    name="v3",
    description="Version 2-3 pipeline (feature concat)",
    add_args=add_task3_v3_args,
    run=lambda args: run_task3_common(args, task3_node_classification_sample_v3, in_dim=1168, load_pretrain_to_gnn=False),
    aliases=("2-3", "sample2-3"),
)

SOC_DIM = register_task(
    name="soc_dim",
    description="Community political polarity prediction (social-dimensions)",
    default_method="baseline",
    add_args=add_soc_dim_args,
    aliases=("social-dimensions", "social_dim"),
)
SOC_DIM.add_method(
    name="baseline",
    description="Binary community polarity classification",
    add_args=lambda p: None,
    run=run_soc_dim_baseline,
    aliases=("default",),
)

COM_CONF = register_task(
    name="com_conf",
    description="Community conflict prediction (user + community features)",
    default_method="baseline",
    add_args=add_com_conf_args,
    aliases=("community_conflict", "community-conflict"),
)
COM_CONF.add_method(
    name="baseline",
    description="Binary conflict classification",
    add_args=lambda p: None,
    run=run_com_conf_baseline,
    aliases=("default",),
)


# -----------------------------
# CLI
# -----------------------------


def _print_registry():
    print("Available tasks (pipeline methods):")
    for task_name, task in TASKS.items():
        alias_text = f" (aliases: {', '.join(task.aliases)})" if task.aliases else ""
        print(f"  - {task_name}{alias_text}: {task.description}")
        for method_name, method in task.methods.items():
            method_aliases = f" (aliases: {', '.join(method.aliases)})" if method.aliases else ""
            print(f"      * {method_name}{method_aliases}: {method.description}")
    print("")
    print("Available model methods (use --conv_name):")
    print("  - " + ", ".join(CONV_NAME_CHOICES))


def _resolve_task_name(name: str) -> str:
    return TASK_ALIASES.get(name, name)


def _resolve_method_name(task_name: str, method_name: Optional[str]) -> Optional[str]:
    if method_name is None:
        return None
    return METHOD_ALIASES.get((task_name, method_name), method_name)


def parse_args(argv: Optional[Iterable[str]] = None):
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--task", type=str, help="Task name. Use --list to see options")
    base.add_argument("--method", type=str, default=None, help="Method name (task-specific)")
    base.add_argument("--list", action="store_true", help="List tasks and methods")
    base.add_argument("--warm_cache", type=str2bool, nargs="?", const=True, default=True,
                      help="Warm user embedding cache before forking workers (recommended).")
    base.add_argument("--mp_start", type=str, default=None, choices=["fork", "spawn", "forkserver"],
                      help="Multiprocessing start method. Use 'spawn' if workers hang.")
    base.add_argument("--data_wait_log_sec", type=int, default=0,
                      help="Log a wait message every N seconds while collecting batches. Set 0 to disable.")
    base.add_argument("--log_file", type=str, default=None,
                      help="Metrics log file path. Default: model_dir + '.log'")
    base.add_argument("--log_disable", type=str2bool, nargs="?", const=True, default=False,
                      help="Disable file logging.")
    base.add_argument("--torch_threads", type=int, default=None,
                      help="Set torch thread count in worker processes (helps avoid CPU oversubscription).")

    base_args, _ = base.parse_known_args(argv)
    if base_args.list:
        _print_registry()
        raise SystemExit(0)

    if not base_args.task:
        _print_registry()
        raise SystemExit("--task is required")

    task_name = _resolve_task_name(base_args.task)
    task = TASKS.get(task_name)
    if task is None:
        _print_registry()
        raise SystemExit(f"Unknown task: {base_args.task}")

    method_name = _resolve_method_name(task.name, base_args.method) or task.default_method
    method = task.methods.get(method_name)
    if method is None:
        _print_registry()
        raise SystemExit(f"Unknown method for task '{task.name}': {base_args.method}")

    parser = argparse.ArgumentParser(
        parents=[base],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Unified pipeline: {task.name}/{method.name}",
    )
    task.add_args(parser)
    method.add_args(parser)

    args = parser.parse_args(argv)
    args.task = task.name
    args.method = method.name
    return args, task, method


def main(argv: Optional[Iterable[str]] = None):
    args, task, method = parse_args(argv)
    if args.mp_start:
        try:
            mp.set_start_method(args.mp_start, force=True)
        except RuntimeError:
            pass
    method.run(args)


if __name__ == "__main__":
    main()
