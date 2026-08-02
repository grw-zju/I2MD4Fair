import torch
import inspect
from collections import Counter, defaultdict
import numpy as np


def recall_(pos_index, pos_len):
    rec_ret = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
    return rec_ret.mean(axis=0)


def ndcg_(pos_index, pos_len):
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float64)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float64)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result.mean(axis=0)


def cal_gini(num_list):
    if len(num_list) <= 1:
        return 0.0
    cum_degree = np.cumsum(sorted(np.append(num_list, 0)))
    sum_degree = cum_degree[-1]
    if sum_degree == 0:
        return 0.0
    xarray = np.array(range(0, len(cum_degree))) / (len(cum_degree) - 1)
    yarray = cum_degree / sum_degree
    b_area = np.trapz(yarray, x=xarray)
    a_area = 0.5 - b_area
    return a_area / (a_area + b_area)


def _get_user_item_embs(model, dataset, device):
    graph_norm = dataset.get_norm_graph().to(device)
    modality_features = dataset.get_modality_features()
    for k in modality_features:
        modality_features[k] = modality_features[k].to(device)

    norm_matrices = dataset.get_modality_norm_matrices()
    inter_norm_u = norm_matrices['inter_norm_u'].to(device)
    inter_norm_v = norm_matrices['inter_norm_v'].to(device)

    model.eval()
    with torch.no_grad():
        if hasattr(model, 'get_user_item_embs'):
            try:
                user_embs, item_embs = model.get_user_item_embs(
                    graph_norm, modality_features,
                    interaction_matrix_norm_u=inter_norm_u,
                    interaction_matrix_norm_v=inter_norm_v
                )
            except TypeError:
                user_embs, item_embs = model.get_user_item_embs(graph_norm, modality_features)
            return user_embs, item_embs
        elif hasattr(model, 'full_sort_predict'):
            return None, None
        elif hasattr(model, 'get_embs'):
            signature = inspect.signature(model.get_embs)
            params = list(signature.parameters.values())
            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            if has_varargs or len(params) >= 2:
                user_embs, item_embs = model.get_embs(graph_norm, modality_features)
            elif len(params) == 1:
                param_name = params[0].name.lower()
                if 'graph' in param_name:
                    user_embs, item_embs = model.get_embs(graph_norm)
                else:
                    user_embs, item_embs = model.get_embs(modality_features)
            else:
                user_embs, item_embs = model.get_embs()
            return user_embs, item_embs
        else:
            raise ValueError("Model must implement get_user_item_embs, full_sort_predict, or get_embs")


def evaluate_model(model, dataset, device='cpu', K_list=[10, 20], mode='test', eval_batch_size=1024):
    n_items = dataset.n_items

    test_user_item_dict = getattr(dataset, 'test_user_item_dict', None)
    if test_user_item_dict is None:
        test_user_item_dict = defaultdict(set)
        for u, v in dataset.test_data:
            test_user_item_dict[u].add(v)

    val_user_item_dict = getattr(dataset, 'val_user_item_dict', None)
    if val_user_item_dict is None:
        val_user_item_dict = defaultdict(set)
        for u, v in dataset.val_data:
            val_user_item_dict[u].add(v)

    train_user_item_dict = dataset.train_user_item_dict
    target_user_item_dict = test_user_item_dict if mode == 'test' else val_user_item_dict
    eval_users = sorted(target_user_item_dict.keys())

    all_topk_indices = []
    recall_hits = {K: 0.0 for K in K_list}
    ndcg_hits = {K: 0.0 for K in K_list}
    total_users = 0
    discount = 1.0 / np.log2(np.arange(2, max(K_list) + 2, dtype=np.float64))

    max_k = max(K_list)
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'full_sort_predict'):
            if hasattr(model, 'prepare_full_sort'):
                model.prepare_full_sort(dataset, device)
            for start in range(0, len(eval_users), eval_batch_size):
                batch_users = eval_users[start:start + eval_batch_size]
                user_tensor = torch.LongTensor(batch_users).to(device)
                batch_scores = model.full_sort_predict(dataset, device, user_tensor)
                _collect_batch_metrics(batch_scores, batch_users, max_k, K_list, n_items,
                                       train_user_item_dict, val_user_item_dict,
                                       target_user_item_dict, mode, all_topk_indices,
                                       recall_hits, ndcg_hits, discount, device)
                total_users += len(batch_users)
            if hasattr(model, 'clear_full_sort_cache'):
                model.clear_full_sort_cache()
        else:
            user_embs, item_embs = _get_user_item_embs(model, dataset, device)
            item_embs_t = item_embs.T
            for start in range(0, len(eval_users), eval_batch_size):
                batch_users = eval_users[start:start + eval_batch_size]
                user_tensor = torch.LongTensor(batch_users).to(device)
                batch_scores = user_embs[user_tensor] @ item_embs_t
                _collect_batch_metrics(batch_scores, batch_users, max_k, K_list, n_items,
                                       train_user_item_dict, val_user_item_dict,
                                       target_user_item_dict, mode, all_topk_indices,
                                       recall_hits, ndcg_hits, discount, device)
                total_users += len(batch_users)

    if total_users == 0:
        return defaultdict(dict)

    topk_index_matrix = np.array(all_topk_indices)

    metrics = defaultdict(dict)
    for K in K_list:
        metrics['NDCG'][K] = round(ndcg_hits[K] / total_users, 4)
        metrics['Recall'][K] = round(recall_hits[K] / total_users, 4)

        topk_at_k = topk_index_matrix[:, :K]
        num_count = Counter(list(topk_at_k.reshape(-1)))
        num_list = np.array(list(num_count.values()), dtype=np.float64)
        p_list = num_list / max(num_list.sum(), 1.0)

        entropy = -np.sum(np.log(p_list + 1e-10) * p_list)
        gini = cal_gini(num_list)
        coverage = len(num_count) / n_items

        metrics['Gini'][K] = round(gini, 4)
        metrics['Entropy'][K] = round(entropy, 4)
        metrics['Coverage'][K] = round(coverage, 4)

    return metrics


def _collect_batch_metrics(batch_scores, batch_users, max_k, K_list, n_items,
                           train_user_item_dict, val_user_item_dict,
                           target_user_item_dict, mode, all_topk_indices,
                           recall_hits, ndcg_hits, discount, device):
    if not torch.is_tensor(batch_scores):
        batch_scores = torch.as_tensor(batch_scores, device=device)
    else:
        batch_scores = batch_scores.to(device)

    mask_rows = []
    mask_cols = []
    for row_idx, u in enumerate(batch_users):
        interacted = train_user_item_dict.get(u, set())
        if mode == 'test':
            interacted = interacted | val_user_item_dict.get(u, set())
        for v in interacted:
            if v < n_items:
                mask_rows.append(row_idx)
                mask_cols.append(v)

    if mask_rows:
        batch_scores[
            torch.as_tensor(mask_rows, device=device, dtype=torch.long),
            torch.as_tensor(mask_cols, device=device, dtype=torch.long)
        ] = -1e10

    _, topk_tensor = torch.topk(batch_scores, k=max_k, dim=1)
    topk_np = topk_tensor.cpu().numpy()
    all_topk_indices.extend(topk_np.tolist())

    for row_idx, u in enumerate(batch_users):
        true_items = target_user_item_dict[u]
        pos_len = max(len(true_items), 1)
        hits = np.fromiter((int(i) in true_items for i in topk_np[row_idx]), dtype=bool, count=max_k)
        for K in K_list:
            hits_k = hits[:K]
            recall_hits[K] += float(hits_k.sum()) / pos_len
            idcg_len = min(pos_len, K)
            idcg = discount[:idcg_len].sum()
            dcg = (hits_k.astype(np.float64) * discount[:K]).sum()
            ndcg_hits[K] += dcg / idcg if idcg > 0 else 0.0
