import torch
import torch.nn as nn
import torch.nn.functional as F


class FREEDOM(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2, knn_k=10,
                 mm_layers=1, mm_image_weight=0.5, dropout=0.1, reg_weight=1e-4):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_ui_layers = n_layers
        self.n_mm_layers = mm_layers
        self.knn_k = knn_k
        self.mm_image_weight = mm_image_weight
        self.dropout = dropout
        self.reg_weight = reg_weight

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.modality_encoders = nn.ModuleDict()
        for k, dim in modality_dims.items():
            self.modality_encoders[k] = nn.Linear(dim, embed_dim)
            nn.init.xavier_uniform_(self.modality_encoders[k].weight)

        self.mm_adj = None
        self.masked_graph = None
        self.precomputed_adj = None

    def _build_knn_graph(self, features, k):
        norm_features = F.normalize(features, dim=1)
        sim = norm_features @ norm_features.T
        _, topk_indices = sim.topk(k, dim=1)
        n = features.shape[0]
        rows = torch.arange(n, device=features.device).unsqueeze(1).expand(-1, k).flatten()
        cols = topk_indices.flatten()
        row_sum = torch.bincount(rows, minlength=n).float().clamp_min(1e-7)
        col_sum = torch.bincount(cols, minlength=n).float().clamp_min(1e-7)
        values = row_sum[rows].pow(-0.5) * col_sum[cols].pow(-0.5)
        return torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n, n)).coalesce()

    def _build_mm_adj(self, modality_features):
        with torch.no_grad():
            keys = list(modality_features.keys())
            if self.precomputed_adj is not None:
                adjs = [adj.to(modality_features[keys[0]].device).coalesce() if adj.is_sparse else adj.to(modality_features[keys[0]].device).to_sparse().coalesce()
                        for adj in self.precomputed_adj]
            else:
                adjs = [self._build_knn_graph(modality_features[k], self.knn_k) for k in keys]
            if len(adjs) == 1:
                self.mm_adj = adjs[0]
            else:
                self.mm_adj = (self.mm_image_weight * adjs[0] +
                               (1.0 - self.mm_image_weight) * adjs[1]).coalesce()

    def set_precomputed_adj(self, adj_matrices):
        adjs = []
        if adj_matrices.get('image_adj') is not None:
            adjs.append(adj_matrices['image_adj'])
        if adj_matrices.get('text_adj') is not None:
            adjs.append(adj_matrices['text_adj'])
        if adjs:
            self.precomputed_adj = [adj.float() for adj in adjs]

    def forward(self, graph_norm, modality_features):
        if self.mm_adj is None or self.mm_adj.device != self.item_emb.weight.device:
            self._build_mm_adj(modality_features)

        h = self.item_emb.weight
        for _ in range(self.n_mm_layers):
            h = torch.sparse.mm(self.mm_adj, h) if self.mm_adj.is_sparse else self.mm_adj @ h

        all_embs = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs_list = [all_embs]
        adj = self._drop_graph_edges(graph_norm) if self.training else graph_norm
        for _ in range(self.n_ui_layers):
            all_embs = torch.sparse.mm(adj, all_embs) if adj.is_sparse else adj @ all_embs
            embs_list.append(all_embs)
        final = torch.mean(torch.stack(embs_list, dim=1), dim=1)
        user_final = final[:self.n_users]
        item_final = final[self.n_users:] + h
        return user_final, item_final

    def _drop_graph_edges(self, graph_norm):
        if self.dropout <= 0 or not graph_norm.is_sparse:
            return graph_norm
        graph_norm = graph_norm.coalesce()
        values = graph_norm.values()
        keep = torch.rand(values.shape, device=values.device) > self.dropout
        if keep.sum() == 0:
            return graph_norm
        return torch.sparse_coo_tensor(
            graph_norm.indices()[:, keep],
            values[keep] / (1.0 - self.dropout),
            graph_norm.shape,
            device=values.device
        ).coalesce()

    def compute_loss(self, user_ids, pos_ids, neg_ids, graph_norm, modality_features):
        user_embs, item_embs = self.forward(graph_norm, modality_features)
        u = user_embs[user_ids]
        pos = item_embs[pos_ids]
        neg = item_embs[neg_ids]
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        modal_loss = torch.tensor(0.0, device=bpr_loss.device)
        for k in modality_features:
            feat = self.modality_encoders[k](modality_features[k])
            modal_loss = modal_loss - F.logsigmoid(
                (u * feat[pos_ids]).sum(dim=1) - (u * feat[neg_ids]).sum(dim=1)
            ).mean()
        return bpr_loss + self.reg_weight * modal_loss + self.reg_weight * self._l2_reg()

    def _l2_reg(self):
        reg = torch.norm(self.user_emb.weight, 2) ** 2 + torch.norm(self.item_emb.weight, 2) ** 2
        for k in self.modality_encoders:
            reg += torch.norm(self.modality_encoders[k].weight, 2) ** 2
        return reg / 2

    def get_embs(self, graph_norm, modality_features):
        return self.forward(graph_norm, modality_features)
