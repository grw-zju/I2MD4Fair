import torch
import torch.nn as nn
import torch.nn.functional as F


class MMGCNBranch(nn.Module):
    def __init__(self, n_users, n_items, feat_dim, embed_dim, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.preference = nn.Parameter(torch.empty(n_users, embed_dim))
        nn.init.xavier_normal_(self.preference)
        self.feature_proj = nn.Linear(feat_dim, embed_dim)
        self.layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)])
        self.gates = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)])
        for layer in list(self.layers) + list(self.gates):
            nn.init.xavier_normal_(layer.weight)

    def forward(self, graph_norm, raw_features, id_embedding):
        item_feat = self.feature_proj(raw_features)
        x = F.normalize(torch.cat([self.preference, item_feat], dim=0), dim=1)
        for layer, gate in zip(self.layers, self.gates):
            h = torch.sparse.mm(graph_norm, x) if graph_norm.is_sparse else graph_norm @ x
            x_hat = F.leaky_relu(layer(x)) + id_embedding
            x = F.leaky_relu(gate(h) + x_hat)
        return x


class MMGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.id_embedding = nn.Parameter(torch.empty(n_users + n_items, embed_dim))
        nn.init.xavier_normal_(self.id_embedding)

        self.branches = nn.ModuleDict()
        for k, dim in modality_dims.items():
            self.branches[k] = MMGCNBranch(n_users, n_items, dim, embed_dim, n_layers=3)

    def forward(self, graph_norm, modality_features):
        representation = None
        num_modal = 0
        for k in modality_features:
            branch_rep = self.branches[k](graph_norm, modality_features[k], self.id_embedding)
            representation = branch_rep if representation is None else representation + branch_rep
            num_modal += 1
        representation = representation / max(num_modal, 1)
        return representation[:self.n_users], representation[self.n_users:]

    def compute_loss(self, user_ids, pos_ids, neg_ids, graph_norm, modality_features):
        user_embs, item_embs = self.forward(graph_norm, modality_features)
        u = user_embs[user_ids]
        pos = item_embs[pos_ids]
        neg = item_embs[neg_ids]
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg = self._l2_reg()
        return bpr_loss + 1e-4 * reg

    def _l2_reg(self):
        reg = torch.norm(self.id_embedding, 2) ** 2
        for branch in self.branches.values():
            reg += torch.norm(branch.preference, 2) ** 2
        return reg / 2

    def get_embs(self, graph_norm, modality_features):
        return self.forward(graph_norm, modality_features)
