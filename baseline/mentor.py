import torch
import torch.nn as nn
import torch.nn.functional as F


class MENTOR(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2,
                 temp=0.2, align_weight=0.1, mask_weight_g=0.1, mask_weight_f=0.01):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.temp = temp
        self.align_weight = align_weight
        self.mask_weight_g = mask_weight_g
        self.mask_weight_f = mask_weight_f

        self.id_item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.id_item_emb.weight)

        self.modality_encoders = nn.ModuleDict()
        self.modality_user_embs = nn.ModuleDict()
        for k, dim in modality_dims.items():
            self.modality_encoders[k] = nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            nn.init.xavier_uniform_(self.modality_encoders[k][0].weight)
            nn.init.xavier_uniform_(self.modality_encoders[k][2].weight)
            self.modality_user_embs[k] = nn.Embedding(n_users, embed_dim)
            nn.init.xavier_uniform_(self.modality_user_embs[k].weight)

        self.mask_rate = 0.1
        self.user_weight = nn.Parameter(torch.empty(n_users, max(len(modality_dims), 1), 1))
        nn.init.xavier_normal_(self.user_weight)
        self.item_weight = nn.Parameter(torch.empty(n_items, max(len(modality_dims), 1), 1))
        nn.init.xavier_normal_(self.item_weight)
        self.mlp = nn.Linear(embed_dim * max(len(modality_dims), 1), embed_dim * max(len(modality_dims), 1))

    def _propagate(self, graph_norm, user_emb, item_emb, perturbed=False):
        all_embs = torch.cat([user_emb, item_emb], dim=0)
        embs_list = [all_embs]
        for _ in range(self.n_layers):
            all_embs = torch.sparse.mm(graph_norm, all_embs) if graph_norm.is_sparse else graph_norm @ all_embs
            if perturbed:
                noise = F.normalize(torch.randn_like(all_embs), dim=-1)
                all_embs = all_embs + torch.sign(all_embs) * noise * 0.1
            embs_list.append(all_embs)
        final = torch.mean(torch.stack(embs_list, dim=1), dim=1)
        return final[:self.n_users], final[self.n_users:]

    def _modality_reps(self, graph_norm, modality_features, perturbed=False):
        user_parts = []
        item_parts = []
        for k in modality_features:
            item_feat = self.modality_encoders[k](modality_features[k])
            u, i = self._propagate(graph_norm, self.modality_user_embs[k].weight, item_feat, perturbed=perturbed)
            user_parts.append(u)
            item_parts.append(i)
        if not user_parts:
            raise ValueError("MENTOR requires at least one modality feature")
        u_stack = torch.stack(user_parts, dim=1)
        i_stack = torch.stack(item_parts, dim=1)
        u_w = F.softmax(self.user_weight[:, :u_stack.size(1)], dim=1)
        i_w = F.softmax(self.item_weight[:, :i_stack.size(1)], dim=1)
        user = (u_w * u_stack).reshape(self.n_users, -1)
        item = (i_w * i_stack).reshape(self.n_items, -1)
        return user, item, user_parts, item_parts

    def forward(self, graph_norm, modality_features):
        user, item, _, _ = self._modality_reps(graph_norm, modality_features, perturbed=False)
        item = item + self._repeat_to_match(self.id_item_emb.weight, item.shape[1])
        return user, item

    def _repeat_to_match(self, emb, dim):
        if emb.shape[1] == dim:
            return emb
        repeats = (dim + emb.shape[1] - 1) // emb.shape[1]
        return emb.repeat(1, repeats)[:, :dim]

    def _info_nce(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        logits = z1 @ z2.T / self.temp
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    def compute_loss(self, user_ids, pos_ids, neg_ids, graph_norm, modality_features):
        user_embs, item_embs = self.forward(graph_norm, modality_features)
        u = user_embs[user_ids]
        pos = item_embs[pos_ids]
        neg = item_embs[neg_ids]
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        user_rep, item_rep, user_parts, item_parts = self._modality_reps(graph_norm, modality_features)
        guide_item = self._repeat_to_match(self.id_item_emb.weight, item_rep.shape[1])
        reps = [item_rep, guide_item] + [self._repeat_to_match(p, item_rep.shape[1]) for p in item_parts]
        means = [r.mean() for r in reps]
        vars_ = [r.var(unbiased=False) for r in reps]
        align_loss = torch.tensor(0.0, device=bpr_loss.device)
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                align_loss = align_loss + (means[i] - means[j]).abs() + (vars_[i] - vars_[j]).abs()

        mask_loss = torch.tensor(0.0, device=bpr_loss.device)
        batch_items = torch.unique(torch.cat([pos_ids, neg_ids]))
        for k in modality_features:
            feat = modality_features[k][batch_items]
            feat_full = self.modality_encoders[k](feat)
            feat_masked = self.modality_encoders[k](
                feat * (torch.rand_like(feat) > self.mask_rate).float()
            )
            mask_loss += 1 - F.cosine_similarity(feat_full, feat_masked, dim=1).mean()

        user_n1, item_n1, _, _ = self._modality_reps(graph_norm, modality_features, perturbed=True)
        user_n2, item_n2, _, _ = self._modality_reps(graph_norm, modality_features, perturbed=True)
        graph_mask_loss = self._info_nce(user_n1[user_ids], user_n2[user_ids]) + self._info_nce(item_n1[pos_ids], item_n2[pos_ids])

        total = (bpr_loss + self.align_weight * align_loss + self.mask_weight_f * mask_loss +
                 self.mask_weight_g * graph_mask_loss + 1e-4 * self._l2_reg())
        return total

    def _l2_reg(self):
        reg = torch.norm(self.id_item_emb.weight, 2) ** 2
        for k in self.modality_encoders:
            for p in self.modality_encoders[k].parameters():
                reg += torch.norm(p, 2) ** 2
        for k in self.modality_user_embs:
            reg += torch.norm(self.modality_user_embs[k].weight, 2) ** 2
        return reg / 2

    def get_embs(self, graph_norm, modality_features):
        return self.forward(graph_norm, modality_features)
