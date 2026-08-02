import torch
import torch.nn as nn
import torch.nn.functional as F


class BM3(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2,
                 dropout=0.2, cl_weight=2.0, reg_weight=1e-4):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.cl_weight = cl_weight
        self.reg_weight = reg_weight

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        self.predictor = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_normal_(self.predictor.weight)

        self.modality_encoders = nn.ModuleDict()
        for k, dim in modality_dims.items():
            self.modality_encoders[k] = nn.Linear(dim, embed_dim)
            nn.init.xavier_normal_(self.modality_encoders[k].weight)

    def forward(self, graph_norm=None, modality_features=None):
        ego = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        if graph_norm is None:
            user_g, item_g = self.user_emb.weight, self.item_emb.weight
        else:
            embs = [ego]
            for _ in range(self.n_layers):
                ego = torch.sparse.mm(graph_norm, ego) if graph_norm.is_sparse else graph_norm @ ego
                embs.append(ego)
            all_embs = torch.stack(embs, dim=1).mean(dim=1)
            user_g, item_g = torch.split(all_embs, [self.n_users, self.n_items], dim=0)
        return user_g, item_g + self.item_emb.weight

    def compute_loss(self, user_ids, pos_ids, neg_ids, *args):
        if len(args) == 1:
            graph_norm, modality_features = None, args[0]
        else:
            graph_norm, modality_features = args

        u_online_ori, i_online_ori = self.forward(graph_norm, modality_features)
        with torch.no_grad():
            u_target = F.dropout(u_online_ori.detach(), self.dropout, training=self.training)
            i_target = F.dropout(i_online_ori.detach(), self.dropout, training=self.training)
        u_online = self.predictor(u_online_ori)[user_ids]
        i_online = self.predictor(i_online_ori)[pos_ids]
        u_target = u_target[user_ids]
        i_target = i_target[pos_ids]

        loss_ui = 1 - F.cosine_similarity(u_online, i_target, dim=-1).mean()
        loss_iu = 1 - F.cosine_similarity(i_online, u_target, dim=-1).mean()
        cl_loss = torch.tensor(0.0, device=u_online.device)
        for k in modality_features:
            feat_online = self.predictor(self.modality_encoders[k](modality_features[k]))[pos_ids]
            feat_target = F.dropout(
                self.modality_encoders[k](modality_features[k]).detach(),
                self.dropout,
                training=self.training
            )[pos_ids]
            cl_loss = cl_loss + 1 - F.cosine_similarity(feat_online, i_target, dim=-1).mean()
            cl_loss = cl_loss + 1 - F.cosine_similarity(feat_online, feat_target, dim=-1).mean()

        return loss_ui + loss_iu + self.cl_weight * cl_loss + self.reg_weight * self._l2_reg()

    def _l2_reg(self):
        reg = torch.norm(self.user_emb.weight, 2) ** 2 + torch.norm(self.item_emb.weight, 2) ** 2
        return reg / 2

    def get_embs(self, *args):
        if len(args) == 1:
            return self.forward(None, args[0])
        return self.forward(args[0], args[1])
