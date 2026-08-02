import torch
import torch.nn as nn
import torch.nn.functional as F


class DMRL(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2,
                 num_factors=2, dropout=0.5, dcor_weight=0.1, reg_weight=0.01):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.num_factors = num_factors
        self.dropout = dropout
        self.dcor_weight = dcor_weight
        self.reg_weight = reg_weight

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

        self.modality_encoders = nn.ModuleDict()
        for k, dim in modality_dims.items():
            encoder = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(dim, 150),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(150, embed_dim),
                nn.LeakyReLU(),
            )
            for m in encoder:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.modality_encoders[k] = encoder

        self.num_modalities = 1 + len(modality_dims)
        self.attention_layers = nn.ModuleList()
        for _ in range(num_factors):
            attn = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear((self.num_modalities + 1) * (embed_dim // num_factors), self.num_modalities),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(self.num_modalities, self.num_modalities, bias=False),
            )
            for m in attn:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.attention_layers.append(attn)

    def _encode_modalities(self, modality_features):
        modality_embs = {}
        for k in modality_features:
            feat = F.normalize(modality_features[k], dim=-1)
            modality_embs[k] = self.modality_encoders[k](feat)
        return modality_embs

    def _split_factors(self, emb):
        chunk_size = self.embed_dim // self.num_factors
        return list(emb.split(chunk_size, dim=-1))

    def _compute_scores(self, user_ids, item_ids, modality_embs):
        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        u_factors = self._split_factors(u_emb)
        i_factors = self._split_factors(i_emb)

        mod_factors = {}
        for k in modality_embs:
            mod_factors[k] = self._split_factors(modality_embs[k][item_ids])

        total_score = torch.zeros(user_ids.shape[0], device=user_ids.device)
        for k_idx in range(self.num_factors):
            cat_input = torch.cat([
                u_factors[k_idx],
                i_factors[k_idx],
                *[mod_factors[k][k_idx] for k in modality_embs]
            ], dim=-1)

            attn_weights = self.attention_layers[k_idx](cat_input)
            attn_weights = F.softmax(attn_weights, dim=-1)

            ui_score = F.softplus((u_factors[k_idx] * i_factors[k_idx]).sum(dim=-1))
            mod_scores = []
            for k in modality_embs:
                mod_scores.append(F.softplus((u_factors[k_idx] * mod_factors[k][k_idx]).sum(dim=-1)))
            all_scores = torch.stack([ui_score] + mod_scores, dim=-1)
            total_score = total_score + (attn_weights * all_scores).sum(dim=-1)

        return total_score

    def _distance_correlation(self, x, y):
        n = x.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=x.device)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        dx = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
        dy = torch.cdist(y.unsqueeze(0), y.unsqueeze(0)).squeeze(0)

        ax = dx - dx.mean(dim=1, keepdim=True) - dx.mean(dim=0, keepdim=True) + dx.mean()
        ay = dy - dy.mean(dim=1, keepdim=True) - dy.mean(dim=0, keepdim=True) + dy.mean()

        dcov = torch.sqrt((ax * ay).sum() / (n * n) + 1e-8)
        dvar_x = torch.sqrt((ax * ax).sum() / (n * n) + 1e-8)
        dvar_y = torch.sqrt((ay * ay).sum() / (n * n) + 1e-8)

        if dvar_x < 1e-8 or dvar_y < 1e-8:
            return torch.tensor(0.0, device=x.device)
        return dcov / (dvar_x * dvar_y + 1e-8)

    def _disentanglement_loss(self, user_ids, item_ids, modality_embs):
        loss = torch.tensor(0.0, device=user_ids.device)
        n_pairs = 0

        u_emb = self.user_embedding(user_ids)
        i_emb = self.item_embedding(item_ids)

        all_embs = [u_emb, i_emb] + [modality_embs[k][item_ids] for k in modality_embs]

        for emb in all_embs:
            factors = self._split_factors(emb)
            for i in range(len(factors)):
                for j in range(i + 1, len(factors)):
                    loss = loss + self._distance_correlation(factors[i], factors[j])
                    n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs
        return loss

    def compute_loss(self, user_ids, pos_ids, neg_ids, *args):
        if len(args) == 1:
            graph_norm, modality_features = None, args[0]
        else:
            graph_norm, modality_features = args

        modality_embs = self._encode_modalities(modality_features)

        pos_scores = self._compute_scores(user_ids, pos_ids, modality_embs)
        neg_scores = self._compute_scores(user_ids, neg_ids, modality_embs)

        bpr_loss = F.softplus(-(pos_scores - neg_scores)).mean()

        dcor_loss = self._disentanglement_loss(user_ids, pos_ids, modality_embs)

        reg_loss = (
            self.user_embedding(user_ids).pow(2).sum() +
            self.item_embedding(torch.unique(torch.cat([pos_ids, neg_ids]))).pow(2).sum()
        ) / 2

        return bpr_loss + self.dcor_weight * dcor_loss + self.reg_weight * reg_loss

    def get_embs(self, *args):
        if len(args) == 1:
            modality_features = args[0]
        else:
            _, modality_features = args

        modality_embs = self._encode_modalities(modality_features)

        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight

        item_cat = i_emb
        for k in modality_embs:
            item_cat = item_cat + modality_embs[k]

        return u_emb, item_cat
