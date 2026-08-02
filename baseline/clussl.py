import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CLUSSL(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims, n_layers=2,
                 n_protos=200, n_cl_layers=2, cl_weight=0.1, reg_weight=0.01):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_protos = n_protos
        self.n_cl_layers = n_cl_layers
        self.cl_weight = cl_weight
        self.reg_weight = reg_weight

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.modality_encoders = nn.ModuleDict()
        self.modality_prototypes = nn.ParameterDict()
        for k, dim in modality_dims.items():
            encoder = nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.LeakyReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for m in encoder:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.modality_encoders[k] = encoder

            proto = torch.empty(n_protos, embed_dim)
            nn.init.xavier_uniform_(proto)
            self.modality_prototypes[k] = nn.Parameter(proto)

        self._proto_graphs = {}

    def _build_proto_graph(self, item_embs, prototypes):
        Z_norm = F.normalize(item_embs, dim=1)
        P_norm = F.normalize(prototypes, dim=1)
        sim = Z_norm @ P_norm.T

        k = min(self.n_protos, item_embs.shape[0])
        topk_val, topk_idx = torch.topk(sim, k, dim=0)
        topk_val = F.softmax(topk_val, dim=0)

        n_items = item_embs.shape[0]
        n_protos = prototypes.shape[0]

        H = torch.zeros(n_items, n_protos, device=item_embs.device)
        rows = topk_idx.flatten()
        cols = torch.arange(n_protos, device=item_embs.device).unsqueeze(1).expand(-1, k).flatten()
        H[rows, cols] = topk_val.flatten()

        D_v = H.sum(dim=1).clamp_min(1e-8)
        D_e = H.sum(dim=0).clamp_min(1e-8)

        D_v_inv_sqrt = torch.pow(D_v, -0.5)
        D_e_inv = torch.pow(D_e, -1)

        item_to_proto = H.T * D_v_inv_sqrt.unsqueeze(0)
        proto_to_item = H * D_e_inv.unsqueeze(0) * D_v_inv_sqrt.unsqueeze(1)

        return item_to_proto, proto_to_item

    def _propagate_proto(self, item_embs, proto_graphs):
        item_to_proto, proto_to_item = proto_graphs
        proto_embs = item_to_proto @ item_embs
        Z = proto_to_item @ proto_embs
        return Z

    def _lightgcn_propagate(self, graph_norm):
        all_embs = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs_list = [all_embs]
        for _ in range(self.n_layers):
            all_embs = torch.sparse.mm(graph_norm, all_embs) if graph_norm.is_sparse else graph_norm @ all_embs
            embs_list.append(all_embs)
        final = torch.mean(torch.stack(embs_list, dim=1), dim=1)
        return final[:self.n_users], final[self.n_users:]

    def _get_modality_item_embs(self, modality_features):
        mod_item_embs = {}
        for k in modality_features:
            mod_item_embs[k] = self.modality_encoders[k](modality_features[k])
        return mod_item_embs

    def compute_loss(self, user_ids, pos_ids, neg_ids, *args):
        if len(args) == 1:
            graph_norm, modality_features = None, args[0]
        else:
            graph_norm, modality_features = args

        u_emb, i_emb = self._lightgcn_propagate(graph_norm) if graph_norm is not None else \
            (self.user_embedding.weight, self.item_embedding.weight)

        mod_item_embs = self._get_modality_item_embs(modality_features)

        enhanced_item = i_emb
        for k in mod_item_embs:
            proto_graphs = self._build_proto_graph(mod_item_embs[k], self.modality_prototypes[k])
            propagated = self._propagate_proto(mod_item_embs[k], proto_graphs)
            enhanced_item = enhanced_item + propagated

        u = u_emb[user_ids]
        pos = enhanced_item[pos_ids]
        neg = enhanced_item[neg_ids]

        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        cl_loss = torch.tensor(0.0, device=u_emb.device)
        mod_keys = list(mod_item_embs.keys())
        if len(mod_keys) >= 2:
            for i in range(len(mod_keys)):
                for j in range(i + 1, len(mod_keys)):
                    k1, k2 = mod_keys[i], mod_keys[j]
                    e1 = F.normalize(mod_item_embs[k1][pos_ids], dim=1)
                    e2 = F.normalize(mod_item_embs[k2][pos_ids], dim=1)
                    pos_sim = (e1 * e2).sum(dim=1)
                    neg_sim = e1 @ e2.T
                    cl_loss = cl_loss - (pos_sim - neg_sim.mean(dim=1)).mean()

        reg_loss = (
            self.user_embedding(user_ids).pow(2).sum() +
            self.item_embedding(torch.unique(torch.cat([pos_ids, neg_ids]))).pow(2).sum()
        ) / 2

        return bpr_loss + self.cl_weight * cl_loss + self.reg_weight * reg_loss

    def get_embs(self, *args):
        if len(args) == 1:
            modality_features = args[0]
            u_emb = self.user_embedding.weight
            i_emb = self.item_embedding.weight
        else:
            graph_norm, modality_features = args
            u_emb, i_emb = self._lightgcn_propagate(graph_norm)

        mod_item_embs = self._get_modality_item_embs(modality_features)

        enhanced_item = i_emb
        for k in mod_item_embs:
            proto_graphs = self._build_proto_graph(mod_item_embs[k], self.modality_prototypes[k])
            propagated = self._propagate_proto(mod_item_embs[k], proto_graphs)
            enhanced_item = enhanced_item + propagated

        return u_emb, enhanced_item
