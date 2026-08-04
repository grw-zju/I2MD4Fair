import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import Counter


class ModalityDebiasingWrapper(nn.Module):
    def __init__(self, base_model, n_users, n_items, embed_dim, modality_dims, model_type='mmssl'):
        super().__init__()
        self.base_model = base_model
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.model_type = model_type

        self._full_sort_cache = None

    def forward(self, graph_norm=None, modality_features=None):
        if graph_norm is not None and modality_features is not None:
            user_embs, item_embs = self.base_model.get_embs(graph_norm, modality_features)
        elif modality_features is not None:
            user_embs, item_embs = self.base_model.get_embs(modality_features)
        else:
            user_embs, item_embs = self.base_model.get_embs(graph_norm)
        return user_embs, item_embs

    def _encode_modality_items(self, k, features, item_ids=None):
        if item_ids is not None:
            selected = features[item_ids]
            if hasattr(self.base_model, 'modality_encoders') and k in self.base_model.modality_encoders:
                return self.base_model.modality_encoders[k](selected)
            if hasattr(self.base_model, 'modality_item_encoders') and k in self.base_model.modality_item_encoders:
                return self.base_model.modality_item_encoders[k](selected)
            if hasattr(self.base_model, 'modality_proj') and k in self.base_model.modality_proj:
                return selected @ self.base_model.modality_proj[k]
            if hasattr(self.base_model, 'branches') and k in self.base_model.branches:
                return self.base_model.branches[k].feature_proj(selected)
            if hasattr(self.base_model, 'content_gcns') and k in self.base_model.content_gcns:
                return self.base_model.content_gcns[k].feature_proj(selected)
            return selected[:, :self.embed_dim] if selected.shape[1] >= self.embed_dim else F.pad(selected, (0, self.embed_dim - selected.shape[1]))

        if hasattr(self.base_model, 'modality_encoders') and k in self.base_model.modality_encoders:
            return self.base_model.modality_encoders[k](features)
        if hasattr(self.base_model, 'modality_item_encoders') and k in self.base_model.modality_item_encoders:
            return self.base_model.modality_item_encoders[k](features)
        if hasattr(self.base_model, 'modality_proj') and k in self.base_model.modality_proj:
            return features @ self.base_model.modality_proj[k]
        if hasattr(self.base_model, 'branches') and k in self.base_model.branches:
            return self.base_model.branches[k].feature_proj(features)
        if hasattr(self.base_model, 'content_gcns') and k in self.base_model.content_gcns:
            return self.base_model.content_gcns[k].feature_proj(features)
        return features[:, :self.embed_dim] if features.shape[1] >= self.embed_dim else F.pad(features, (0, self.embed_dim - features.shape[1]))

    def compute_loss(self, user_ids, pos_ids, neg_ids, graph_norm, modality_features):
        import inspect
        sig = inspect.signature(self.base_model.compute_loss)
        n_params = len(sig.parameters)
        if n_params >= 5:
            return self.base_model.compute_loss(user_ids, pos_ids, neg_ids, graph_norm, modality_features)
        else:
            return self.base_model.compute_loss(user_ids, pos_ids, neg_ids, modality_features) if n_params == 4 \
                else self.base_model.compute_loss(user_ids, pos_ids, neg_ids, graph_norm)

    def get_embs(self, graph_norm, modality_features):
        return self.forward(graph_norm, modality_features)

    def full_sort_predict(self, dataset, device, user_ids=None):
        if self._full_sort_cache is None:
            self.prepare_full_sort(dataset, device)
        user_embs, item_embs, modality_items, cf_items, rank_weights = self._full_sort_cache
        if user_ids is None:
            user_ids = torch.arange(self.n_users, device=device)
        else:
            user_ids = user_ids.to(device)

        scores = user_embs[user_ids] @ item_embs.T
        gates = []
        score_stars = []
        for k in modality_items:
            mod_items = self._match_item_dim(modality_items[k], user_embs.shape[1])
            cf_mod_items = self._match_item_dim(cf_items[k], user_embs.shape[1])
            mod_score = user_embs[user_ids] @ mod_items.T
            gates.append(torch.sigmoid(mod_score))
            score_stars.append(user_embs[user_ids] @ cf_mod_items.T)
        if len(gates) < 2:
            return scores

        joint_gate = torch.ones_like(scores)
        for gate in gates:
            joint_gate = joint_gate * gate
        debiased = scores * joint_gate
        for k, score_star in zip(modality_items.keys(), score_stars):
            weight = self._user_item_rank_weight(
                user_embs[user_ids], self._match_item_dim(modality_items[k], user_embs.shape[1]), rank_weights[k]
            )
            debiased = debiased - weight * score_star * joint_gate
        return debiased

    def prepare_full_sort(self, dataset, device):
        graph_norm = dataset.get_norm_graph().to(device)
        modality_features = dataset.get_modality_features()
        for k in modality_features:
            modality_features[k] = modality_features[k].to(device)

        user_embs, item_embs = self.forward(graph_norm, modality_features)

        modality_items = {}
        cf_items = {}
        rank_weights = {}
        for k in modality_features:
            modality_items[k] = self._encode_modality_items(k, modality_features[k])
            mean_features = modality_features[k].mean(dim=0, keepdim=True).expand_as(modality_features[k])
            cf_items[k] = self._encode_modality_items(k, mean_features)
            rank_weights[k] = self._get_modality_rank_weights(dataset, k, modality_features[k]).to(device)
        self._full_sort_cache = (user_embs, item_embs, modality_items, cf_items, rank_weights)

    def clear_full_sort_cache(self):
        self._full_sort_cache = None

    def _user_item_rank_weight(self, user_embs, modality_item_embs, item_rank_weight):
        scores = user_embs @ modality_item_embs.T
        ranks = torch.argsort(torch.argsort(-scores, dim=1), dim=1).float().clamp(max=99)
        item_rank = item_rank_weight.unsqueeze(0).expand_as(ranks)
        k = self._rank_temperature()
        return torch.exp(-k * torch.abs(ranks - item_rank))

    def _match_item_dim(self, item_embs, target_dim):
        if item_embs.shape[1] == target_dim:
            return item_embs
        if item_embs.shape[1] > target_dim:
            return item_embs[:, :target_dim]
        repeats = (target_dim + item_embs.shape[1] - 1) // item_embs.shape[1]
        return item_embs.repeat(1, repeats)[:, :target_dim]

    def _rank_temperature(self):
        if self.model_type == 'lgmrec':
            return 0.0015
        if self.model_type == 'diffmm':
            return 0.5
        if self.model_type == 'mentor':
            return 0.5
        if self.model_type == 'mmssl':
            return 0.0015
        return 0.5

    def _get_modality_rank_weights(self, dataset, modality_key, features):
        ds_dir = os.path.join(dataset.data_dir, dataset.dataset_name)
        cache_path = os.path.join(ds_dir, f'_md_{modality_key}_rank.npy')
        if os.path.exists(cache_path):
            rank = np.load(cache_path).astype(np.float32)
            if rank.shape[0] == self.n_items:
                return torch.from_numpy(rank)

        labels = self._cluster_modality_features(ds_dir, modality_key, features)
        inter_items = dataset.train_data[:, 1] if dataset.train_data.size else np.arange(self.n_items)
        label_counter = Counter(labels[inter_items])
        label_order = [label for label, _ in sorted(label_counter.items(), key=lambda x: x[1], reverse=True)]
        fallback_rank = len(label_order)
        label_rank = {label: idx for idx, label in enumerate(label_order)}
        rank = np.array([label_rank.get(label, fallback_rank) for label in labels], dtype=np.float32)
        rank = np.clip(rank, 0, 99)
        np.save(cache_path, rank)
        return torch.from_numpy(rank)

    def _cluster_modality_features(self, ds_dir, modality_key, features, n_clusters=100):
        label_path = os.path.join(ds_dir, f'_md_{modality_key}_labels.npy')
        if os.path.exists(label_path):
            labels = np.load(label_path).astype(np.int64)
            if labels.shape[0] == self.n_items:
                return labels

        x = features.detach().cpu().numpy().astype(np.float32)
        k = min(n_clusters, self.n_items)
        try:
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(x)
        except Exception:
            labels = np.arange(self.n_items, dtype=np.int64)
        np.save(label_path, labels)
        return labels
