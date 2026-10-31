import torch
import torch.nn as nn
import torch.nn.functional as F


class VBPR(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, visual_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.user_emb = nn.Embedding(n_users, embed_dim * 2)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.visual_proj = nn.Linear(visual_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.visual_proj.weight)

    def _extract_visual(self, modality_features):
        if isinstance(modality_features, dict):
            return modality_features.get('visual', modality_features.get(list(modality_features.keys())[0]))
        return modality_features

    def forward(self, modality_features):
        visual_features = self._extract_visual(modality_features)
        item_visual = self.visual_proj(visual_features)
        item_embs = torch.cat([self.item_emb.weight, item_visual], dim=1)
        user_embs = self.user_emb.weight
        return user_embs, item_embs

    def compute_loss(self, user_ids, pos_ids, neg_ids, modality_features):
        user_embs, item_embs = self.forward(modality_features)
        u = user_embs[user_ids]
        pos = item_embs[pos_ids]
        neg = item_embs[neg_ids]
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg = (torch.norm(self.user_emb.weight, 2) ** 2 +
               torch.norm(self.item_emb.weight, 2) ** 2 +
               torch.norm(self.visual_proj.weight, 2) ** 2) / 2
        return bpr_loss + 1e-4 * reg

    def get_embs(self, modality_features):
        return self.forward(modality_features)
