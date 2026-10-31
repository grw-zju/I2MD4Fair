import torch
import torch.nn as nn
import torch.nn.functional as F


class HGNNLayer(nn.Module):
    def __init__(self, n_hyper_layer):
        super().__init__()
        self.n_hyper_layer = n_hyper_layer

    def forward(self, item_hyper, user_hyper, item_embeds):
        item_ret = item_embeds
        user_ret = None
        for _ in range(self.n_hyper_layer):
            latent = item_hyper.T @ item_ret
            item_ret = item_hyper @ latent
            user_ret = user_hyper @ latent
        return user_ret, item_ret


class LGMRec(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2, n_protos=64,
                 n_mm_layers=2, n_hyper_layers=1, keep_rate=0.5, alpha=0.3,
                 cl_weight=1e-4, reg_weight=1e-6, tau=0.2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_mm_layers = n_mm_layers
        self.n_hyper_layers = n_hyper_layers
        self.n_protos = n_protos
        self.keep_rate = keep_rate
        self.alpha = alpha
        self.cl_weight = cl_weight
        self.reg_weight = reg_weight
        self.tau = tau

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.modality_proj = nn.ParameterDict()
        self.hyper_proj = nn.ParameterDict()
        for k, dim in modality_dims.items():
            self.modality_proj[k] = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(dim, embed_dim)))
            self.hyper_proj[k] = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(dim, n_protos)))

        self.drop = nn.Dropout(p=1 - keep_rate)
        self.hgnn = HGNNLayer(n_hyper_layers)

    def _ui_adj(self, graph_norm):
        return graph_norm.coalesce() if graph_norm.is_sparse else graph_norm

    def _interaction_adj(self, graph_norm):
        graph_norm = graph_norm.coalesce()
        idx = graph_norm.indices()
        vals = graph_norm.values()
        mask = (idx[0] < self.n_users) & (idx[1] >= self.n_users)
        ui_idx = torch.stack([idx[0, mask], idx[1, mask] - self.n_users])
        return torch.sparse_coo_tensor(ui_idx, vals[mask], (self.n_users, self.n_items), device=vals.device).coalesce()

    def _cge(self, graph_norm):
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [ego]
        adj = self._ui_adj(graph_norm)
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(adj, ego) if adj.is_sparse else adj @ ego
            embs.append(ego)
        return torch.stack(embs, dim=1).mean(dim=1)

    def _mge(self, graph_norm, ui_adj, item_feats):
        user_feats = torch.sparse.mm(ui_adj, item_feats) if ui_adj.is_sparse else ui_adj @ item_feats
        deg = torch.sparse.sum(ui_adj, dim=1).to_dense().clamp_min(1e-7).unsqueeze(1) if ui_adj.is_sparse else ui_adj.sum(1, keepdim=True).clamp_min(1e-7)
        user_feats = user_feats / deg
        feats = torch.cat([user_feats, item_feats], dim=0)
        adj = self._ui_adj(graph_norm)
        for _ in range(self.n_mm_layers):
            feats = torch.sparse.mm(adj, feats) if adj.is_sparse else adj @ feats
        return feats

    def forward(self, graph_norm, modality_features):
        cge = self._cge(graph_norm)
        ui_adj = self._interaction_adj(graph_norm)
        hyper_embeddings = []
        mge_sum = torch.zeros_like(cge)
        ghe_sum = torch.zeros_like(cge)
        for k in modality_features:
            raw = modality_features[k]
            item_feats = raw @ self.modality_proj[k]
            item_hyper = F.gumbel_softmax(raw @ self.hyper_proj[k], tau=self.tau, dim=1, hard=False)
            user_hyper = torch.sparse.mm(ui_adj, item_hyper) if ui_adj.is_sparse else ui_adj @ item_hyper
            mge_sum = mge_sum + F.normalize(self._mge(graph_norm, ui_adj, item_feats), dim=1)
            u_h, i_h = self.hgnn(self.drop(item_hyper), self.drop(user_hyper), cge[self.n_users:])
            hyper_embeddings.append((u_h, i_h))
            ghe_sum = ghe_sum + torch.cat([u_h, i_h], dim=0)
        all_embs = cge if not hyper_embeddings else cge + mge_sum + self.alpha * F.normalize(ghe_sum, dim=1)
        user_embs, item_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)
        return user_embs, item_embs, hyper_embeddings

    def compute_loss(self, user_ids, pos_ids, neg_ids, graph_norm, modality_features):
        user_embs, item_embs, hyper_embeddings = self.forward(graph_norm, modality_features)
        u = user_embs[user_ids]
        pos = item_embs[pos_ids]
        neg = item_embs[neg_ids]
        bpr_loss = -F.logsigmoid((u * pos).sum(dim=1) - (u * neg).sum(dim=1)).mean()

        hcl_loss = torch.tensor(0.0, device=bpr_loss.device)
        if len(hyper_embeddings) >= 2:
            (u1, i1), (u2, i2) = hyper_embeddings[:2]
            hcl_loss = self._ssl_triple_loss(u1[user_ids], u2[user_ids], u2)
            hcl_loss = hcl_loss + self._ssl_triple_loss(i1[pos_ids], i2[pos_ids], i2)
        reg = (u.norm(2) + pos.norm(2) + neg.norm(2)) / user_ids.shape[0]
        return bpr_loss + self.cl_weight * hcl_loss + self.reg_weight * reg

    def _ssl_triple_loss(self, emb1, emb2, all_emb):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        all_emb = F.normalize(all_emb, dim=1)
        pos = torch.exp((emb1 * emb2).sum(dim=1) / self.tau)
        ttl = torch.exp(emb1 @ all_emb.T / self.tau).sum(dim=1).clamp_min(1e-8)
        return -torch.log(pos / ttl).sum()

    def _l2_reg(self):
        reg = torch.norm(self.user_emb.weight, 2) ** 2 + torch.norm(self.item_emb.weight, 2) ** 2
        for k in self.modality_proj:
            reg += torch.norm(self.modality_proj[k], 2) ** 2
            reg += torch.norm(self.hyper_proj[k], 2) ** 2
        return reg / 2

    def get_embs(self, graph_norm, modality_features):
        user_embs, item_embs, _ = self.forward(graph_norm, modality_features)
        return user_embs, item_embs
