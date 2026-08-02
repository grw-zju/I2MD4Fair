import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffMM(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2, n_diff_steps=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_diff_steps = n_diff_steps

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

        self.diffusion_nets = nn.ModuleDict()
        for k, dim in modality_dims.items():
            self.diffusion_nets[k] = nn.ModuleList()
            for _ in range(n_diff_steps):
                self.diffusion_nets[k].append(nn.Linear(embed_dim, embed_dim))

    def _diffusion_refine(self, feat, diff_net):
        if not self.training:
            return feat
        t = torch.randint(0, self.n_diff_steps, (1,)).item()
        noise = torch.randn_like(feat) * 0.1
        noisy_feat = feat + noise
        h = noisy_feat
        for i in range(t + 1):
            h = diff_net[i](h)
        return feat + 0.1 * (h - feat)

    def forward(self, graph_norm, modality_features):
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight

        modality_embs = {}
        for k in modality_features:
            feat = self.modality_encoders[k](modality_features[k])
            refined = self._diffusion_refine(feat, self.diffusion_nets[k])
            modality_embs[k] = refined

        enhanced_item = item_emb
        for k in modality_embs:
            enhanced_item = enhanced_item + modality_embs[k]

        all_embs = torch.cat([user_emb, enhanced_item], dim=0)
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

        diff_loss = torch.tensor(0.0, device=bpr_loss.device)
        batch_items = torch.unique(torch.cat([pos_ids, neg_ids]))
        for k in modality_features:
            feat = self.modality_encoders[k](modality_features[k][batch_items])
            noise = torch.randn_like(feat) * 0.1
            noisy = feat + noise
            for step in range(self.n_diff_steps):
                pred = self.diffusion_nets[k][step](noisy)
                diff_loss += F.mse_loss(pred, feat)

        total = bpr_loss + 0.1 * contrastive_loss + 0.01 * diff_loss + 1e-4 * self._l2_reg()
        return total

    def _l2_reg(self):
        reg = torch.norm(self.user_emb.weight, 2) ** 2 + torch.norm(self.item_emb.weight, 2) ** 2
        for k in self.modality_encoders:
            for p in self.modality_encoders[k].parameters():
                reg += torch.norm(p, 2) ** 2
        return reg / 2

    def get_embs(self, graph_norm, modality_features):
        return self.forward(graph_norm, modality_features)
