import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, graph_norm):
        all_embs = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs_list = [all_embs]
        for _ in range(self.n_layers):
            all_embs = torch.sparse.mm(graph_norm, all_embs) if graph_norm.is_sparse else graph_norm @ all_embs
            embs_list.append(all_embs)
        final_embs = torch.mean(torch.stack(embs_list, dim=1), dim=1)
        user_embs = final_embs[:self.n_users]
        item_embs = final_embs[self.n_users:]
        return user_embs, item_embs

    def compute_loss(self, user_ids, pos_ids, neg_ids, graph_norm):
        user_embs, item_embs = self.forward(graph_norm)
        u = user_embs[user_ids]
        pos = item_embs[pos_ids]
        neg = item_embs[neg_ids]
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg = (u.norm(2).pow(2) + pos.norm(2).pow(2) + neg.norm(2).pow(2)) / (2 * user_ids.shape[0])
        return loss + 1e-4 * reg

    def get_embs(self, graph_norm):
        return self.forward(graph_norm)

    def _l2_reg(self):
        reg = torch.norm(self.user_emb.weight, 2) ** 2 + torch.norm(self.item_emb.weight, 2) ** 2
        return reg / 2


class LightGCNWithModalities(LightGCN):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2):
        super().__init__(n_users, n_items, embed_dim, n_layers)
        self.modality_item_encoders = nn.ModuleDict()
        for k, dim in modality_dims.items():
            self.modality_item_encoders[k] = nn.Linear(dim, embed_dim)
            nn.init.xavier_uniform_(self.modality_item_encoders[k].weight)

    def get_embs_with_modalities(self, graph_norm, modality_features):
        user_embs, item_embs = self.forward(graph_norm)
        modality_embs = {}
        for k in modality_features:
            modality_embs[k] = self.modality_item_encoders[k](modality_features[k])
        hat_X_U = user_embs
        hat_X_V = item_embs
        for k in modality_embs:
            hat_X_V = torch.cat([hat_X_V, modality_embs[k]], dim=1)
        return hat_X_U, hat_X_V
