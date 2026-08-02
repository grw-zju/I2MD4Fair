import torch
import torch.nn as nn
import torch.nn.functional as F


class MMSSL(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.modality_encoders = nn.ModuleDict()
        for k, dim in modality_dims.items():
            self.modality_encoders[k] = nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            nn.init.xavier_uniform_(self.modality_encoders[k][0].weight)
            nn.init.xavier_uniform_(self.modality_encoders[k][2].weight)

        self.modality_user_embs = nn.ModuleDict()
        for k, dim in modality_dims.items():
            self.modality_user_embs[k] = nn.Embedding(n_users, embed_dim)
            nn.init.xavier_uniform_(self.modality_user_embs[k].weight)

        self.adversarial_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, graph_norm, modality_features):
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight

        Z_u_dict = {}
        Z_v_dict = {}
        for k in modality_features:
            Z_v_k = self.modality_encoders[k](modality_features[k])
            Z_u_k = self.modality_user_embs[k].weight
            Z_u_dict[k] = Z_u_k
            Z_v_dict[k] = Z_v_k

        modality_item = item_emb
        for k in Z_v_dict:
            modality_item = modality_item + Z_v_dict[k]

        modality_user = user_emb
        for k in Z_u_dict:
            modality_user = modality_user + Z_u_dict[k]

        all_embs = torch.cat([modality_user, modality_item], dim=0)
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

        contrastive_loss = torch.tensor(0.0, device=bpr_loss.device)
        modality_keys = list(modality_features.keys())
        if len(modality_keys) >= 2:
            batch_items = torch.unique(torch.cat([pos_ids, neg_ids]))
            for i in range(len(modality_keys)):
                for j in range(i + 1, len(modality_keys)):
                    k1, k2 = modality_keys[i], modality_keys[j]
                    z1 = F.normalize(self.modality_encoders[k1](modality_features[k1])[batch_items], dim=1)
                    z2 = F.normalize(self.modality_encoders[k2](modality_features[k2])[batch_items], dim=1)
                    sim = z1 @ z2.T / 0.01
                    labels = torch.arange(sim.shape[0], device=sim.device)
                    contrastive_loss += F.cross_entropy(sim, labels)

        adv_loss = torch.tensor(0.0, device=bpr_loss.device)
        batch_items = torch.unique(torch.cat([pos_ids, neg_ids]))
        for k in modality_features:
            z_k = self.modality_encoders[k](modality_features[k][batch_items])
            pred = self.adversarial_net(z_k)
            adv_loss += F.binary_cross_entropy_with_logits(pred, torch.ones_like(pred) * 0.5)

        total = bpr_loss + 0.1 * contrastive_loss + 0.01 * adv_loss + 1e-4 * self._l2_reg()
        return total

    def _l2_reg(self):
        reg = torch.norm(self.user_emb.weight, 2) ** 2 + torch.norm(self.item_emb.weight, 2) ** 2
        for k in self.modality_encoders:
            for p in self.modality_encoders[k].parameters():
                reg += torch.norm(p, 2) ** 2
        for k in self.modality_user_embs:
            reg += torch.norm(self.modality_user_embs[k].weight, 2) ** 2
        return reg / 2

    def get_embs(self, graph_norm, modality_features):
        return self.forward(graph_norm, modality_features)
