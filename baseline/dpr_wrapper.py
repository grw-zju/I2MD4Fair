import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class DPRWrapper(nn.Module):
    def __init__(self, base_model, n_users, n_items, embed_dim, modality_dims,
                 n_groups=5, alpha=1.0, beta=1.0):
        super().__init__()
        self.base_model = base_model
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_groups = n_groups
        self.alpha = alpha
        self.beta = beta

        self._full_sort_cache = None
        self._item_group_weights = None

    def compute_loss(self, user_ids, pos_ids, neg_ids, *args):
        if len(args) == 1:
            graph_norm, modality_features = None, args[0]
        else:
            graph_norm, modality_features = args

        base_loss = self.base_model.compute_loss(user_ids, pos_ids, neg_ids, *args)

        if self._item_group_weights is not None:
            device = base_loss.device
            pos_weights = self._item_group_weights[pos_ids]
            neg_weights = self._item_group_weights[neg_ids]
            fairness_reg = (pos_weights - neg_weights).mean()
            return base_loss + self.alpha * fairness_reg

        return base_loss

    def _compute_group_weights(self, dataset):
        if dataset.train_data.size:
            item_freq = np.bincount(dataset.train_data[:, 1], minlength=self.n_items).astype(np.float32)[:self.n_items]
        else:
            item_freq = np.zeros(self.n_items, dtype=np.float32)

        sorted_indices = np.argsort(-item_freq)
        group_size = self.n_items // self.n_groups
        weights = np.ones(self.n_items, dtype=np.float32)

        for g in range(self.n_groups):
            start = g * group_size
            end = start + group_size if g < self.n_groups - 1 else self.n_items
            mask = sorted_indices[start:end]
            weights[mask] = (g + 1) / self.n_groups

        weights = torch.from_numpy(weights)
        weights = weights / weights.mean()

        return weights

    def get_embs(self, *args):
        return self.base_model.get_embs(*args)

    def get_user_item_embs(self, *args, **kwargs):
        if hasattr(self.base_model, 'get_user_item_embs'):
            return self.base_model.get_user_item_embs(*args, **kwargs)
        u, i = self.base_model.get_embs(*args)
        return u, i

    def prepare_full_sort(self, dataset, device):
        if self._item_group_weights is None:
            self._item_group_weights = self._compute_group_weights(dataset).to(device)

        if hasattr(self.base_model, 'prepare_full_sort'):
            self.base_model.prepare_full_sort(dataset, device)

        self._full_sort_cache = self._get_embs_safe(dataset, device)

    def _get_embs_safe(self, dataset, device):
        graph_norm = dataset.get_norm_graph().to(device)
        modality_features = dataset.get_modality_features()
        for k in modality_features:
            modality_features[k] = modality_features[k].to(device)
        return self.base_model.get_embs(graph_norm, modality_features)

    def full_sort_predict(self, dataset, device, user_ids=None):
        if self._full_sort_cache is None:
            self.prepare_full_sort(dataset, device)
        user_embs, item_embs = self._full_sort_cache

        if user_ids is None:
            user_ids = torch.arange(self.n_users, device=device)
        else:
            user_ids = user_ids.to(device)

        scores = user_embs[user_ids] @ item_embs.T

        weights = self._item_group_weights.to(device)
        scores = scores * weights.unsqueeze(0)

        return scores

    def clear_full_sort_cache(self):
        self._full_sort_cache = None
        if hasattr(self.base_model, 'clear_full_sort_cache'):
            self.base_model.clear_full_sort_cache()
