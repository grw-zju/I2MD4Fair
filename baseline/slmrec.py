import torch
import torch.nn as nn
import torch.nn.functional as F


class SLMRec(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2,
                 temp=0.2, ssl_temp=0.2, ssl_alpha=0.1, dropout_rate=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.temp = temp
        self.ssl_temp = ssl_temp
        self.ssl_alpha = ssl_alpha
        self.dropout_rate = dropout_rate

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.modality_encoders = nn.ModuleDict()
        for k, dim in modality_dims.items():
            self.modality_encoders[k] = nn.Linear(dim, embed_dim)
            nn.init.xavier_uniform_(self.modality_encoders[k].weight)

        fusion_dim = embed_dim * (len(modality_dims) + 1)
        self.user_fusion = nn.Linear(fusion_dim, embed_dim)
        self.item_fusion = nn.Linear(fusion_dim, embed_dim)
        nn.init.xavier_uniform_(self.user_fusion.weight)
        nn.init.xavier_uniform_(self.item_fusion.weight)

    def _propagate(self, graph_norm, user_emb, item_emb):
        all_embs = torch.cat([user_emb, item_emb], dim=0)
        embs_list = [all_embs]
        for _ in range(self.n_layers):
            all_embs = torch.sparse.mm(graph_norm, all_embs) if graph_norm.is_sparse else graph_norm @ all_embs
            embs_list.append(all_embs)
        final = torch.mean(torch.stack(embs_list, dim=1), dim=1)
        return final[:self.n_users], final[self.n_users:]

    def forward(self, graph_norm, modality_features):
        user_parts = []
        item_parts = []
        u_id, i_id = self._propagate(graph_norm, self.user_emb.weight, self.item_emb.weight)
        user_parts.append(u_id)
        item_parts.append(i_id)
        for k in modality_features:
            feat = F.normalize(self.modality_encoders[k](F.normalize(modality_features[k], dim=1)), dim=1)
            u_m, i_m = self._propagate(graph_norm, self.user_emb.weight, feat)
            user_parts.append(u_m)
            item_parts.append(i_m)
        return self.user_fusion(torch.cat(user_parts, dim=1)), self.item_fusion(torch.cat(item_parts, dim=1))

    def compute_loss(self, user_ids, pos_ids, neg_ids, graph_norm, modality_features):
        user_embs, item_embs = self.forward(graph_norm, modality_features)
        u = F.normalize(user_embs[user_ids], dim=1)
        pos = F.normalize(item_embs[pos_ids], dim=1)
        logits = u @ pos.T / self.temp
        labels = torch.arange(logits.shape[0], device=logits.device)
        main_loss = F.cross_entropy(logits, labels)

        ssl_loss = torch.tensor(0.0, device=main_loss.device)
        batch_items = torch.unique(pos_ids)
        modality_keys = list(modality_features.keys())
        if len(modality_keys) >= 2:
            for i in range(len(modality_keys)):
                for j in range(i + 1, len(modality_keys)):
                    k1, k2 = modality_keys[i], modality_keys[j]
                    z1 = F.normalize(self.modality_encoders[k1](modality_features[k1])[batch_items], dim=1)
                    z2 = F.normalize(self.modality_encoders[k2](modality_features[k2])[batch_items], dim=1)
                    sim = z1 @ z2.T / self.ssl_temp
                    labels = torch.arange(sim.shape[0], device=sim.device)
                    ssl_loss += F.cross_entropy(sim, labels)
        for k in modality_features:
            feat = modality_features[k][batch_items]
            masked = feat * (torch.rand_like(feat) > self.dropout_rate).float()
            z1 = F.normalize(self.modality_encoders[k](feat), dim=1)
            z2 = F.normalize(self.modality_encoders[k](masked), dim=1)
            sim = z1 @ z2.T / self.ssl_temp
            labels = torch.arange(sim.shape[0], device=sim.device)
            ssl_loss += F.cross_entropy(sim, labels)

        return main_loss + self.ssl_alpha * ssl_loss + 1e-4 * self._l2_reg()

    def _l2_reg(self):
        reg = torch.norm(self.user_emb.weight, 2) ** 2 + torch.norm(self.item_emb.weight, 2) ** 2
        for k in self.modality_encoders:
            for p in self.modality_encoders[k].parameters():
                reg += torch.norm(p, 2) ** 2
        return reg / 2

    def get_embs(self, graph_norm, modality_features):
        return self.forward(graph_norm, modality_features)
