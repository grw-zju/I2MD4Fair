import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentGCN(nn.Module):
    def __init__(self, n_users, feat_dim, embed_dim, n_routing=2):
        super().__init__()
        self.n_users = n_users
        self.n_routing = n_routing
        self.preference = nn.Parameter(torch.empty(n_users, embed_dim))
        nn.init.xavier_normal_(self.preference)
        self.feature_proj = nn.Linear(feat_dim, embed_dim)
        nn.init.xavier_normal_(self.feature_proj.weight)

    def forward(self, graph_norm, raw_features):
        item_feat = F.normalize(F.leaky_relu(self.feature_proj(raw_features)), dim=1)
        preference = F.normalize(self.preference, dim=1)
        for _ in range(self.n_routing):
            x = torch.cat([preference, item_feat], dim=0)
            h = torch.sparse.mm(graph_norm, x) if graph_norm.is_sparse else graph_norm @ x
            preference = F.normalize(preference + h[:self.n_users], dim=1)
        x = torch.cat([preference, item_feat], dim=0)
        h = torch.sparse.mm(graph_norm, x) if graph_norm.is_sparse else graph_norm @ x
        return x + h


class GRCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.id_embedding = nn.Parameter(torch.empty(n_users + n_items, embed_dim))
        nn.init.xavier_normal_(self.id_embedding)

        self.content_gcns = nn.ModuleDict()
        for k, dim in modality_dims.items():
            self.content_gcns[k] = ContentGCN(n_users, dim, embed_dim, n_routing=n_layers)

        self.model_specific_conf = nn.Parameter(torch.empty(n_users + n_items, len(modality_dims)))
        nn.init.xavier_normal_(self.model_specific_conf)
        self.id_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(2)])
        for layer in self.id_layers:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, graph_norm, modality_features):
        content_parts = []
        modal_weights = []
        idx = 0
        for k in modality_features:
            rep = self.content_gcns[k](graph_norm, modality_features[k])
            content_parts.append(rep)
            conf = torch.sigmoid(self.model_specific_conf[:, idx:idx + 1])
            modal_weights.append(conf)
            idx += 1
        content_rep = torch.cat(content_parts, dim=1)
        weight = torch.stack(modal_weights, dim=0).max(dim=0).values.clamp_min(0.0)

        id_rep = F.normalize(self.id_embedding, dim=1)
        for layer in self.id_layers:
            h = torch.sparse.mm(graph_norm, id_rep) if graph_norm.is_sparse else graph_norm @ id_rep
            id_rep = id_rep + F.leaky_relu(layer(h * weight))
        representation = torch.cat([id_rep, content_rep], dim=1)
        return representation[:self.n_users], representation[self.n_users:]

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
        reg = torch.norm(self.id_embedding, 2) ** 2 + torch.norm(self.model_specific_conf, 2) ** 2
        for k in self.content_gcns:
            for p in self.content_gcns[k].parameters():
                reg += torch.norm(p, 2) ** 2
        return reg / 2

    def get_embs(self, graph_norm, modality_features):
        return self.forward(graph_norm, modality_features)
