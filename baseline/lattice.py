import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class LATTICE(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2, knn_k=10):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.knn_k = knn_k

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.modality_encoders = nn.ModuleDict()
        self.item_graph_layers = nn.ModuleDict()
        self.precomputed_adj = None
        for k, dim in modality_dims.items():
            self.modality_encoders[k] = nn.Linear(dim, embed_dim)
            nn.init.xavier_uniform_(self.modality_encoders[k].weight)
            self.item_graph_layers[k] = nn.ModuleList()
            for _ in range(n_layers):
                self.item_graph_layers[k].append(nn.Linear(embed_dim, embed_dim))
                nn.init.xavier_uniform_(self.item_graph_layers[k][-1].weight)

    def _build_item_graph(self, features, k):
        norm_features = F.normalize(features, dim=1)
        sim = norm_features @ norm_features.T
        _, topk_indices = sim.topk(k, dim=1)
        n = features.shape[0]
        rows = torch.arange(n).unsqueeze(1).expand(-1, k).flatten()
        cols = topk_indices.flatten()
        values = torch.ones(rows.shape[0], device=features.device)
        adj = torch.sparse.FloatTensor(
            torch.stack([rows, cols]), values, (n, n)
        ).to_dense()
        adj = adj + adj.T
        adj = adj * (1 - torch.eye(n, device=features.device))
        degree = adj.sum(dim=1, keepdim=True) + 1e-8
        norm_adj = adj / degree
        return norm_adj

    def set_precomputed_adj(self, adj_matrices):
        adjs = []
        if adj_matrices.get('image_adj') is not None:
            adjs.append(adj_matrices['image_adj'])
        if adj_matrices.get('text_adj') is not None:
            adjs.append(adj_matrices['text_adj'])
        if adjs:
            self.precomputed_adj = [adj.float() for adj in adjs]

    def forward(self, graph_norm, modality_features):
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight

        item_semantic_embs = {}
        for k in modality_features:
            M_k = modality_features[k]
            feat = self.modality_encoders[k](M_k)
            if self.precomputed_adj is not None and len(self.precomputed_adj) > 0:
                adj = self.precomputed_adj[min(len(item_semantic_embs), len(self.precomputed_adj) - 1)].to(feat.device)
            else:
                adj = self._build_item_graph(feat, self.knn_k)
            h = feat
            for layer in self.item_graph_layers[k]:
                if adj.is_sparse:
                    propagated = torch.sparse.mm(adj, h)
                else:
                    propagated = adj @ h
                h = F.leaky_relu(propagated + layer(h), 0.2)
            item_semantic_embs[k] = h

        modality_item_emb = item_emb
        for k in item_semantic_embs:
            modality_item_emb = modality_item_emb + item_semantic_embs[k]

        all_embs = torch.cat([user_emb, modality_item_emb], dim=0)
        embs_list = [all_embs]
        for _ in range(self.n_layers):
            all_embs = torch.sparse.mm(graph_norm, all_embs) if graph_norm.is_sparse else graph_norm @ all_embs
            embs_list.append(all_embs)
        final = torch.mean(torch.stack(embs_list, dim=1), dim=1)
        user_final = final[:self.n_users]
        item_final = final[self.n_users:]
        return user_final, item_final

    def compute_loss(self, user_ids, pos_ids, neg_ids, graph_norm, modality_features):
        user_embs, item_embs = self.forward(graph_norm, modality_features)
        u = user_embs[user_ids]
        pos = item_embs[pos_ids]
        neg = item_embs[neg_ids]
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return bpr_loss + 1e-4 * self._l2_reg()

    def _l2_reg(self):
        reg = torch.norm(self.user_emb.weight, 2) ** 2 + torch.norm(self.item_emb.weight, 2) ** 2
        for k in self.modality_encoders:
            reg += torch.norm(self.modality_encoders[k].weight, 2) ** 2
        return reg / 2

    def get_embs(self, graph_norm, modality_features):
        return self.forward(graph_norm, modality_features)
