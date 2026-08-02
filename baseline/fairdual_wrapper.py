import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class FairDualWrapper(nn.Module):
    def __init__(self, base_model, n_users, n_items, embed_dim, modality_dims,
                 n_groups=5, eta=0.01, alpha=0.8, topk=20):
        super().__init__()
        self.base_model = base_model
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_groups = n_groups
        self.eta = eta
        self.alpha = alpha
        self.topk = topk

        self._full_sort_cache = None
        self._item_group_ids = None
        self._group_mu = None

    def compute_loss(self, user_ids, pos_ids, neg_ids, *args):
        if len(args) == 1:
            graph_norm, modality_features = None, args[0]
        else:
            graph_norm, modality_features = args

        base_loss = self.base_model.compute_loss(user_ids, pos_ids, neg_ids, *args)

        if self._item_group_ids is not None and self._group_mu is not None:
            device = base_loss.device
            group_ids = self._item_group_ids.to(device)
            mu = self._group_mu.to(device)

            pos_group_weights = mu[group_ids[pos_ids]]
            neg_group_weights = mu[group_ids[neg_ids]]

            fairness_reg = pos_group_weights.mean() - neg_group_weights.mean()
            return base_loss + self.eta * fairness_reg

        return base_loss

    def _compute_item_groups(self, dataset):
        item_freq = np.zeros(self.n_items, dtype=np.int64)
        for u, i in dataset.train_data:
            if i < self.n_items:
                item_freq[i] += 1

        sorted_indices = np.argsort(-item_freq, kind='stable')
        group_size = self.n_items // self.n_groups
        group_ids = np.zeros(self.n_items, dtype=np.int64)

        for g in range(self.n_groups):
            start = g * group_size
            end = start + group_size if g < self.n_groups - 1 else self.n_items
            mask = sorted_indices[start:end]
            group_ids[mask] = g

        return torch.from_numpy(group_ids)

    def update_dual_variables(self, dataset, device, epoch, update_every=5):
        if epoch % update_every != 0:
            return

        if self._item_group_ids is None:
            self._item_group_ids = self._compute_item_groups(dataset)

        if self._full_sort_cache is None:
            self.prepare_full_sort(dataset, device)

        user_embs, item_embs = self._full_sort_cache
        scores = user_embs @ item_embs.T

        train_mask = torch.ones(self.n_users, self.n_items, device=device)
        for u in range(self.n_users):
            interacted = dataset.train_user_item_dict.get(u, set())
            for v in interacted:
                if v < self.n_items:
                    train_mask[u, v] = 0

        scores = scores * train_mask - 1e10 * (1 - train_mask)
        _, topk_idx = torch.topk(scores, self.topk, dim=1)

        group_ids = self._item_group_ids.to(device)
        group_exposure = torch.zeros(self.n_groups, device=device)
        for g in range(self.n_groups):
            group_mask = (group_ids == g)
            group_exposure[g] = group_mask[topk_idx].sum().float()

        min_exposure = group_exposure.min()
        max_exposure = group_exposure.max()

        if max_exposure > min_exposure:
            group_exposure_norm = group_exposure / max_exposure
            target_weight = min_exposure / (group_exposure + 1e-8)
            target_weight = target_weight / target_weight.mean()

            if self._group_mu is None:
                self._group_mu = torch.ones(self.n_groups, device=device)

            grad_mu = target_weight - self._group_mu
            self._group_mu = self.alpha * self._group_mu + (1 - self.alpha) * (self._group_mu + self.eta * grad_mu)
            self._group_mu = F.softplus(self._group_mu)

    def get_embs(self, *args):
        return self.base_model.get_embs(*args)

    def get_user_item_embs(self, *args, **kwargs):
        if hasattr(self.base_model, 'get_user_item_embs'):
            return self.base_model.get_user_item_embs(*args, **kwargs)
        u, i = self.base_model.get_embs(*args)
        return u, i

    def prepare_full_sort(self, dataset, device):
        if hasattr(self.base_model, 'prepare_full_sort'):
            self.base_model.prepare_full_sort(dataset, device)

        if hasattr(self.base_model, 'get_user_item_embs'):
            user_embs, item_embs = self.base_model.get_user_item_embs(
                dataset.get_norm_graph().to(device),
                {k: v.to(device) for k, v in dataset.get_modality_features().items()}
            )
        else:
            graph_norm = dataset.get_norm_graph().to(device)
            modality_features = dataset.get_modality_features()
            for k in modality_features:
                modality_features[k] = modality_features[k].to(device)
            user_embs, item_embs = self.base_model.get_embs(graph_norm, modality_features)

        self._full_sort_cache = (user_embs, item_embs)

    def full_sort_predict(self, dataset, device, user_ids=None):
        if self._full_sort_cache is None:
            self.prepare_full_sort(dataset, device)
        user_embs, item_embs = self._full_sort_cache

        if user_ids is None:
            user_ids = torch.arange(self.n_users, device=device)
        else:
            user_ids = user_ids.to(device)

        scores = user_embs[user_ids] @ item_embs.T

        if self._item_group_ids is not None and self._group_mu is not None:
            group_ids = self._item_group_ids.to(device)
            mu = self._group_mu.to(device)
            item_weights = mu[group_ids]
            scores = scores * item_weights.unsqueeze(0)

        return scores

    def clear_full_sort_cache(self):
        self._full_sort_cache = None
        if hasattr(self.base_model, 'clear_full_sort_cache'):
            self.base_model.clear_full_sort_cache()
