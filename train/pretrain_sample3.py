# -*- coding: UTF-8 -*-
# @Date    ：2023/3/18
# @Author  ：Zhang Xinnong
# @File    ：pretrain_sample2.py
"""
this is a DIY version of pretraining:
(1) pretrain user-graph only
repeat following step iteratively
(2) pooling to get sub embedding
(3) pretrain sub node
"""

import sys
import os
import hashlib
import tempfile
# 把 community_encoder 加入 path，使 train/ 下脚本能 import GPT_GNN（位于上级目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GPT_GNN.data import *
from GPT_GNN.model import *
from GPT_GNN.utils import _get_user_embed_cache
from warnings import filterwarnings
import argparse
import matplotlib.pyplot as plt


filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Pre-training GNN on a given graph (heterogeneous / homogeneous)')
'''
   GPT-GNN arguments 
'''
parser.add_argument('--attr_ratio', type=float, default=0.5,
                    help='Ratio of attr-loss against link-loss, range: [0-1]')
parser.add_argument('--attr_type', type=str, default='vec',
                    choices=['text', 'vec'],
                    help='The type of attribute decoder')
parser.add_argument('--neg_samp_num', type=int, default=255,
                    help='Maximum number of negative sample for each target node.')
parser.add_argument('--queue_size', type=int, default=256,
                    help='Max size of adaptive embedding queue.')
parser.add_argument('--w2v_dir', type=str, default='./dataset/w2v_all',
                    help='The address of preprocessed graph.')
'''
    Dataset arguments
'''
parser.add_argument('--data_dir', type=str, default='../sample/graphs_sample.pk',
                    help='The address of preprocessed graph.')
parser.add_argument('--pretrain_model_dir', type=str, default='../models/model3_epoch10l_stage1.pt',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--pretrain_model_dir2', type=str, default='../models/model3_epoch10l_stage2.pt',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--current_model_dir', type=str, default='../models/model3_epoch10l_current.pt',
                    help='The address for storing the models and optimization results.')
parser.add_argument('--stage_ckpt_dir', type=str, default='',
                    help='Folder to store stage-switch checkpoints (saved after each stage).')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--sample_depth', type=int, default=4,  # 6
                    help='How many layers within a mini-batch subgraph')
parser.add_argument('--sample_width', type=int, default=64,  # 128
                    help='How many nodes to be sampled per layer per type')

'''
   Model arguments 
'''
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=400,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=3,
                    help='Number of GNN layers')
parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers', action='store_true')
parser.add_argument('--dropout', type=int, default=0.2,
                    help='Dropout ratio')

'''
    Optimization arguments
'''
parser.add_argument('--max_lr', type=float, default=1e-3,
                    help='Maximum learning rate.')
parser.add_argument('--scheduler', type=str, default='cosine',  # cycle by default
                    help='Name of learning rate scheduler.', choices=['cycle', 'cosine'])
parser.add_argument('--n_epoch', type=int, default=10,
                    help='Number of epoch to run')
parser.add_argument('--n_iteration', type=int, default=12,
                    help='Number of iteration to run')
parser.add_argument('--n_pool', type=int, default=4,  # 8 by default
                    help='Number of process to sample subgraph')
parser.add_argument('--n_batch', type=int, default=16,  # 32 by default
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--batch_size', type=int, default=128,  # 256 by default
                    help='Number of output nodes for training')
parser.add_argument('--clip', type=float, default=0.5,
                    help='Gradient Norm Clipping')

parser.add_argument('--stage', type=int, default=1,
                    help='pretrain stage')
parser.add_argument('--keep_pretrain', type=bool, default=False,
                    help='keep pretrain from models')

args = parser.parse_args()

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

graph = renamed_load(open(args.data_dir, 'rb'))

def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _prepare_stage_ckpt_dir(dir_path: str) -> str:
    if not dir_path:
        return ''
    target_dir = os.path.abspath(dir_path)
    os.makedirs(target_dir, exist_ok=True)
    test_path = os.path.join(target_dir, "stage1_iter1.pt")
    if len(test_path) <= 240:
        return target_dir
    link_root = os.path.join(tempfile.gettempdir(), "community_encoder_ckpt_links")
    os.makedirs(link_root, exist_ok=True)
    digest = hashlib.sha1(target_dir.encode("utf-8")).hexdigest()[:10]
    link_path = os.path.join(link_root, f"ckpt_{digest}")
    if os.path.islink(link_path):
        return link_path
    if os.path.exists(link_path):
        link_path = os.path.join(link_root, f"ckpt_{digest}_{os.getpid()}")
    try:
        os.symlink(target_dir, link_path)
    except OSError as e:
        print(f"[WARN] stage_ckpt_dir path too long and symlink failed ({e}); will try original path.")
        return target_dir
    return link_path

def _is_path_too_long_error(err: Exception) -> bool:
    msg = str(err)
    return "File name too long" in msg or (isinstance(err, OSError) and getattr(err, "errno", None) == 36)

def _safe_save_ckpt(state_dict, path: str, fallback_dir: str = ''):
    try:
        torch.save(state_dict, path)
        return path, False
    except (RuntimeError, OSError) as e:
        print(f"[DEBUG] ckpt save failed: {e}")
        print(f"[DEBUG] ckpt path len={len(path)} basename_len={len(os.path.basename(path))} path={path}")
        if not _is_path_too_long_error(e):
            raise
        if fallback_dir:
            os.makedirs(fallback_dir, exist_ok=True)
            short_path = os.path.join(fallback_dir, os.path.basename(path))
            print(f"[DEBUG] ckpt fallback_dir len={len(fallback_dir)} path={fallback_dir}")
            print(f"[DEBUG] ckpt short_path len={len(short_path)} basename_len={len(os.path.basename(short_path))} path={short_path}")
            try:
                torch.save(state_dict, short_path)
                return short_path, True
            except (RuntimeError, OSError) as e2:
                print(f"[DEBUG] ckpt fallback save failed: {e2}")
                if not _is_path_too_long_error(e2):
                    raise
        # 最后兜底：写入极短路径，避免作业中断
        tiny_root = os.path.join(tempfile.gettempdir(), "ce_ckpt")
        os.makedirs(tiny_root, exist_ok=True)
        digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
        tiny_path = os.path.join(tiny_root, f"ckpt_{digest}.pt")
        print(f"[DEBUG] ckpt tiny_path len={len(tiny_path)} basename_len={len(os.path.basename(tiny_path))} path={tiny_path}")
        try:
            torch.save(state_dict, tiny_path)
            return tiny_path, True
        except (RuntimeError, OSError) as e3:
            print(f"[DEBUG] ckpt tiny save failed: {e3}")
            if _is_path_too_long_error(e3):
                print(f"[WARN] stage ckpt path too long; skipping save. last path tried: {tiny_path}")
                return None, True
            raise

def GPT_sample(seed, target_nodes, time_range, batch_size, feature_extractor, target_type, rel_stop_list):
    np.random.seed(seed)
    samp_target_nodes = target_nodes[np.random.choice(len(target_nodes), batch_size)]
    threshold = 0.5
    feature, times, edge_list, _, attr = sample_subgraph(graph, time_range,
                                                         inp={target_type: samp_target_nodes},
                                                         feature_extractor=feature_extractor,
                                                         sampled_depth=args.sample_depth,
                                                         sampled_number=args.sample_width)
    rem_edge_list = defaultdict(  # source_type
        lambda: defaultdict(  # relation_type
            lambda: []  # [target_id, source_id]
        ))

    ori_list = {}
    for source_type in edge_list[target_type]:
        ori_list[source_type] = {}
        for relation_type in edge_list[target_type][source_type]:
            ori_list[source_type][relation_type] = np.array(edge_list[target_type][source_type][relation_type])
            el = []
            for target_ser, source_ser in edge_list[target_type][source_type][relation_type]:
                # print(target_ser, source_ser) # idx, idx
                if relation_type not in rel_stop_list and target_ser < batch_size and \
                        np.random.random() > threshold:
                    rem_edge_list[source_type][relation_type] += [[target_ser, source_ser]]
                    continue
                el += [[target_ser, source_ser]]
                # el += [[source_ser, target_ser]]
            el = np.array(el)
            edge_list[target_type][source_type][relation_type] = el

            if relation_type == 'self':
                continue
            # else:
            #     if 'rev_' in relation_type:
            #         rev_relation = relation_type[4:]
            #     else:
            #         rev_relation = 'rev_' + relation_type
            #     edge_list[source_type]['paper'][rev_relation] = list(np.stack((el[:, 1], el[:, 0])).T)

    '''
        Adding feature nodes:
    '''
    n_target_nodes = len(feature[target_type])
    feature[target_type] = np.concatenate((feature[target_type], np.zeros([batch_size, feature[target_type].shape[1]])))
    times[target_type] = np.concatenate((times[target_type], times[target_type][:batch_size]))

    for source_type in edge_list[target_type]:
        for relation_type in edge_list[target_type][source_type]:
            el = []
            for target_ser, source_ser in edge_list[target_type][source_type][relation_type]:
                if target_ser < batch_size:
                    if relation_type == 'self':
                        el += [[target_ser + n_target_nodes, target_ser + n_target_nodes]]
                    else:
                        el += [[target_ser + n_target_nodes, source_ser]]
            if len(el) > 0:
                edge_list[target_type][source_type][relation_type] = \
                    np.concatenate((edge_list[target_type][source_type][relation_type], el))

    rem_edge_lists = {}
    for source_type in rem_edge_list:
        rem_edge_lists[source_type] = {}
        for relation_type in rem_edge_list[source_type]:
            rem_edge_lists[source_type][relation_type] = np.array(rem_edge_list[source_type][relation_type])
    del rem_edge_list

    return to_torch(feature, times, edge_list, graph), rem_edge_lists, ori_list, attr[:batch_size], \
           (n_target_nodes, n_target_nodes + batch_size)


def prepare_data(pool):
    jobs = []
    for _ in np.arange(args.n_batch - 1):
        jobs.append(
            pool.apply_async(GPT_sample, args=(randint(), pre_target_nodes, {1: True}, args.batch_size, feature_OAG, target_type, rel_stop_list)))
    jobs.append(
        pool.apply_async(GPT_sample, args=(randint(), train_target_nodes, {1: True}, args.batch_size, feature_OAG, target_type, rel_stop_list)))
    return jobs


if __name__ == '__main__':
    loss_records1 = []
    loss_records2 = []
    best_val1 = 10000   # -3.314;  -3.614;  -3.636;  -3.606;  -3.453
    best_val2 = 10000   # -2.289;  -2.547;  -2.722;  -2.713;  -2.619
    best_epoch1 = 0
    best_epoch2 = 0
    best_iteration1 = 0
    best_iteration2 = 0

    # 预热用户嵌入缓存，fork 之后可共享内存页，避免 stage2 重新加载 10GB
    if args.n_pool > 0:
        _get_user_embed_cache()

    # 复用进程池，避免每个 iteration 重建导致子进程重复加载大缓存
    pool = mp.Pool(args.n_pool)
    stage_ckpt_dir = _prepare_stage_ckpt_dir(args.stage_ckpt_dir)
    stage_ckpt_fallback_dir = ''
    if args.stage_ckpt_dir:
        digest = hashlib.sha1(os.path.abspath(args.stage_ckpt_dir).encode("utf-8")).hexdigest()[:10]
        stage_ckpt_fallback_dir = os.path.join(tempfile.gettempdir(), f"community_encoder_ckpts_{digest}")
    if stage_ckpt_dir and stage_ckpt_dir != os.path.abspath(args.stage_ckpt_dir):
        print(f"Stage ckpt dir too long; using symlink path: {stage_ckpt_dir}")

    try:
        for iter_idx in range(args.n_iteration):
            if args.stage == 1:
                target_type = 'user'
                rel_stop_list = ['self', 'sub_sub']
                # rel_stop_list = ['self', 'sub_sub', 'sub_user']
            else:
                target_type = 'sub'
                rel_stop_list = ['self', 'user_user']
                # rel_stop_list = ['self', 'user_user', 'sub_user']
    
            idx = np.arange(len(graph.node_feature[target_type]))
    
            np.random.seed(43)
            np.random.shuffle(idx)
    
            pre_idx = idx[: int(len(idx) * 0.7)]
            train_idx = idx[int(len(idx) * 0.7): int(len(idx) * 0.8)]
            valid_idx = idx[int(len(idx) * 0.8): int(len(idx) * 0.9)]
            test_idx = idx[int(len(idx) * 0.9):]
    
            pre_target_nodes = np.concatenate([pre_idx, np.ones(len(pre_idx))]).reshape(2, -1).transpose()
            train_target_nodes = np.concatenate([train_idx, np.ones(len(train_idx))]).reshape(2, -1).transpose()
    
            st = time.time()
            jobs = prepare_data(pool)
            repeat_num = int(len(pre_target_nodes) / args.batch_size // args.n_batch)
            data, rem_edge_list, ori_edge_list, _, _ = GPT_sample(randint(), pre_target_nodes, {1: True}, args.batch_size,
                                                                  feature_OAG, target_type, rel_stop_list)
            node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data
            types = graph.get_types()
    
            gnn = GNN(conv_name=args.conv_name,
                      # in_dim=len(graph.node_feature[target_type]['embed'].values[0]),
                      in_dim=len(graph.node_feature['user']['embed'].values[0]),
                      n_hid=args.n_hid,
                      n_heads=args.n_heads,
                      n_layers=args.n_layers,
                      dropout=args.dropout,
                      num_types=len(types),
                      num_relations=len(graph.get_meta_graph()) + 1,
                      prev_norm=args.prev_norm,
                      last_norm=args.last_norm)
    
            attr_decoder = Matcher(gnn.n_hid, gnn.in_dim)
    
            gpt_gnn = GPT_GNN(gnn=gnn, rem_edge_list=rem_edge_list, attr_decoder=attr_decoder,
                              types=types, neg_samp_num=args.neg_samp_num, device=device)
            gpt_gnn.init_emb.data = node_feature[node_type == node_dict[target_type][1]].mean(dim=0).detach()
    
            print("Iteration: ", iter_idx + 1)
            if iter_idx > 0:
                print('loading parameters from existing models..Current stage: ', args.stage)
                gpt_gnn.load_state_dict(torch.load(args.current_model_dir))
                # if args.stage == 1:
                #     gpt_gnn.load_state_dict(torch.load(args.pretrain_model_dir2))
                # else:
                #     gpt_gnn.load_state_dict(torch.load(args.pretrain_model_dir))
    
            gpt_gnn = gpt_gnn.to(device)
    
            if args.stage == 1:
                best_val = best_val1
                best_epoch = best_epoch1
            else:
                best_val = best_val2
                best_epoch = best_epoch2
    
            train_step = 0
            stats = []
            loss_records = []
    
            optimizer = torch.optim.AdamW(gpt_gnn.parameters(), weight_decay=1e-2, eps=1e-06, lr=args.max_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, repeat_num * args.n_batch, eta_min=1e-6)
    
            print('Start Pretraining...')
            # if args.stage == 1:
            #     args.n_epoch = 10
            # else:
            #     args.n_epoch = 20
    
            for epoch in np.arange(args.n_epoch) + 1:
                gpt_gnn.neg_queue_size = args.queue_size * epoch // args.n_epoch
                for batch in np.arange(repeat_num) + 1:
                    # print("Epoch: %d, (%d / %d)" % (epoch, batch, repeat_num))
                    train_data = [job.get() for job in jobs[:-1]]
                    valid_data = jobs[-1].get()
                    # 复用进程池，不再每次重建（避免子进程重复加载10GB缓存）
                    jobs = prepare_data(pool)
                    et = time.time()
                    # print('Data Preparation: %.1fs' % (et - st))
    
                    train_link_losses = []
                    train_attr_losses = []
                    gpt_gnn.train()
                    for data, rem_edge_list, ori_edge_list, attr, (start_idx, end_idx) in train_data:
                        node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data
                        node_feature = node_feature.detach()
                        node_feature[start_idx: end_idx] = gpt_gnn.init_emb
                        node_emb = gpt_gnn.gnn(node_feature.to(device), node_type.to(device), edge_time.to(device),
                                               edge_index.to(device), edge_type.to(device))
    
                        loss_link, _ = gpt_gnn.link_loss(node_emb, rem_edge_list, ori_edge_list, node_dict, target_type,
                                                         use_queue=True, update_queue=True)
                        # print(node_emb[start_idx: end_idx], torch.FloatTensor(attr).to(device))
                        # print(len(node_emb[start_idx: end_idx]), len(attr))     # 256:batch_size, 192
                        loss_attr = gpt_gnn.feat_loss(node_emb[start_idx: end_idx], torch.FloatTensor(attr).to(device))
    
                        # TODO check
                        if args.stage == 1:
                            loss = loss_link + loss_attr * args.attr_ratio
                        else:
                            loss = loss_link + loss_attr * args.attr_ratio
    
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(gpt_gnn.parameters(), args.clip)
                        optimizer.step()
    
                        train_link_losses += [loss_link.item()]
                        train_attr_losses += [loss_attr.item()]
                        scheduler.step()
                        scheduler.step()
                    '''
                        Valid
                    '''
                    gpt_gnn.eval()
                    with torch.no_grad():
                        data, rem_edge_list, ori_edge_list, attr, (start_idx, end_idx) = valid_data
                        node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = data
                        node_feature = node_feature.detach()
                        node_feature[start_idx: end_idx] = gpt_gnn.init_emb
                        node_emb = gpt_gnn.gnn(node_feature.to(device), node_type.to(device), edge_time.to(device),
                                               edge_index.to(device), edge_type.to(device))
                        loss_link, ress = gpt_gnn.link_loss(node_emb, rem_edge_list, ori_edge_list, node_dict, target_type,
                                                            use_queue=False, update_queue=True)
                        loss_link = loss_link.item()
                        loss_attr = gpt_gnn.feat_loss(node_emb[start_idx: end_idx], torch.FloatTensor(attr).to(device))
    
                        ndcgs = []
                        for res in ress:
                            ai = np.zeros(len(res[0]))
                            ai[0] = 1
                            ndcgs += [ndcg_at_k(ai[j.cpu().numpy()], len(j)) for j in res.argsort(descending=True)]
    
                        # TODO check
                        if args.stage == 1:
                            valid_loss = loss_link + loss_attr * args.attr_ratio
                        else:
                            valid_loss = loss_link + loss_attr * args.attr_ratio
    
                        st = time.time()
                        valid_loss_cpu = valid_loss.cpu()
                        # loss_records.append(valid_loss_cpu)
                        if args.stage == 1:
                            loss_records1.append(valid_loss_cpu)
                        else:
                            loss_records2.append(valid_loss_cpu)
    
                        print((
                                  "Stage: %d  Epoch: %d, (%d / %d) %.1fs  LR: %.5f Train Loss: (%.3f, %.3f) = %.3f  Valid Loss: (%.3f, %.3f) = %.3f  NDCG: %.3f  Norm: %.3f  queue: %d") %
                              (args.stage, epoch, batch, repeat_num, (st - et), optimizer.param_groups[0]['lr'],
                               np.average(train_link_losses),
                               np.average(train_attr_losses), loss, loss_link, loss_attr, valid_loss, np.average(ndcgs),
                               node_emb.norm(dim=1).mean(),
                               gpt_gnn.neg_queue_size))
    
                    if valid_loss < best_val:
                        best_val = valid_loss
                        best_epoch = epoch
                        print('UPDATE!!!')
                        if args.stage == 1:
                            _ensure_parent_dir(args.pretrain_model_dir)
                            torch.save(gpt_gnn.state_dict(), args.pretrain_model_dir)
                            best_val1 = best_val
                            best_epoch1 = best_epoch
                            best_iteration1 = iter_idx + 1
                        else:
                            _ensure_parent_dir(args.pretrain_model_dir2)
                            torch.save(gpt_gnn.state_dict(), args.pretrain_model_dir2)
                            best_val2 = best_val
                            best_epoch2 = best_epoch
                            best_iteration2 = iter_idx + 1
    
                    stats += [[np.average(train_link_losses), loss_link, loss_attr, valid_loss]]
    
            _ensure_parent_dir(args.current_model_dir)
            torch.save(gpt_gnn.state_dict(), args.current_model_dir)
            if stage_ckpt_dir:
                stage_ckpt_path = os.path.join(stage_ckpt_dir, f"stage{args.stage}_iter{iter_idx + 1}.pt")
                print(f"[DEBUG] stage_ckpt_path len={len(stage_ckpt_path)} basename_len={len(os.path.basename(stage_ckpt_path))} path={stage_ckpt_path}")
                saved_path, used_fallback = _safe_save_ckpt(
                    gpt_gnn.state_dict(),
                    stage_ckpt_path,
                    stage_ckpt_fallback_dir,
                )
                if saved_path is None:
                    print(f"[WARN] stage ckpt not saved; path too long: {stage_ckpt_path}")
                elif used_fallback:
                    print(f"[WARN] stage ckpt path too long; saved to {saved_path} instead of {stage_ckpt_path}")
                else:
                    print(f"Stage-switch checkpoint saved: {saved_path}")
    
            if args.stage == 1:
                args.stage = 2
            else:
                args.stage = 1
            print('Stage Pretraining done! Best epoch: %d, best valid loss: %.3f' % (best_epoch, best_val))
        
    finally:
        # 训练完成后关闭进程池
        pool.close()
        pool.join()

    print('Pretraining done!\nStage1 Best epoch: %d, best valid loss: %.3f\nStage2 Best epoch: %d, best valid loss: %.3f' %
          (best_epoch1, best_val1, best_epoch2, best_val2))
    pretrain_round1 = list(range(len(loss_records1)))
    pretrain_round2 = list(range(len(loss_records2)))
    print(loss_records1)
    print('\n')
    print(loss_records2)
    # loss_records.to(torch.device("cpu"))
    plt.figure(1)
    plt.subplot(211)
    plt.plot(pretrain_round1, loss_records1)
    plt.subplot(212)
    plt.plot(pretrain_round2, loss_records2)
    plt.show()
