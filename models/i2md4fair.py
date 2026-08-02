import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinkhornOT(nn.Module):
    def __init__(self, eps=0.1, n_iters=3):
        super().__init__()
        self.eps = eps
        self.n_iters = n_iters

    def forward(self, cost_matrix):
        K = torch.exp(-cost_matrix / self.eps).clamp_min(1e-12)
        n_items, n_protos = K.shape
        a = torch.ones(n_items, device=K.device) / n_items
        b = torch.ones(n_protos, device=K.device) / n_protos
        for _ in range(self.n_iters):
            a = 1.0 / (n_items * (K @ b).clamp_min(1e-12))
            b = 1.0 / (n_protos * (K.T @ a).clamp_min(1e-12))
        gamma = a.unsqueeze(1) * K * b.unsqueeze(0)
        return gamma


class SoftPrototypeClustering(nn.Module):
    def __init__(self, n_items, n_protos, embed_dim, eps=0.1):
        super().__init__()
        prototypes = torch.empty(n_protos, embed_dim)
        nn.init.xavier_uniform_(prototypes)
        self.prototypes = nn.Parameter(prototypes)
        self.ot_solver = SinkhornOT(eps=eps, n_iters=3)

    def forward(self, Z_v):
        Z_norm = F.normalize(Z_v, dim=1)
        P_norm = F.normalize(self.prototypes, dim=1)
        cost = 1 - Z_norm @ P_norm.T
        gamma = self.ot_solver(cost)
        return gamma


class HypergraphConv(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W = nn.Parameter(torch.empty(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, Z_v, incidence_matrix):
        H = incidence_matrix
        D_v = H.sum(dim=1).clamp_min(1e-8)
        D_e = H.sum(dim=0).clamp_min(1e-8)
        Z_tmp = Z_v * torch.pow(D_v, -0.5).unsqueeze(1)
        Z_tmp = H.T @ Z_tmp
        Z_tmp = Z_tmp * torch.pow(D_e, -1).unsqueeze(1)
        Z_tmp = H @ Z_tmp
        Z_tmp = Z_tmp * torch.pow(D_v, -0.5).unsqueeze(1)
        Z_out = Z_tmp @ self.W
        return F.relu(Z_out)


class IntraMDM(nn.Module):
    def __init__(self, n_items, n_protos, embed_dim, eps=0.1):
        super().__init__()
        self.n_items = n_items
        self.soft_clustering = SoftPrototypeClustering(n_items, n_protos, embed_dim, eps)
        self.hgcn = HypergraphConv(embed_dim)

    def forward(self, Z_v):
        gamma = self.soft_clustering(Z_v)
        B_I = Z_v.shape[0]
        incidence = B_I * gamma
        Z_v_debiased = self.hgcn(Z_v, incidence)
        return Z_v_debiased, gamma


class CLUBEstimator(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim * 2),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        params = self.net(x)
        mu, logvar = params.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, -10, 10)
        return mu, logvar

    def log_prob(self, z, mu, logvar):
        return -0.5 * (logvar + (z - mu) ** 2 / (torch.exp(logvar) + 1e-8))

    def mi_estimate(self, modality_input, debiased_emb):
        with torch.no_grad():
            mu, logvar = self.forward(modality_input)
        log_pos = self.log_prob(debiased_emb, mu, logvar).sum(dim=-1)
        shuffled = debiased_emb[torch.randperm(debiased_emb.shape[0])]
        log_neg = self.log_prob(shuffled, mu, logvar).sum(dim=-1)
        mi_est = (log_pos - log_neg).mean()
        return torch.clamp(mi_est, min=0)

    def nll_loss(self, modality_input, debiased_emb):
        mu, logvar = self.forward(modality_input)
        nll = -self.log_prob(debiased_emb, mu, logvar).mean()
        return nll


class InterMDM(nn.Module):
    def __init__(self, modality_dims, embed_dim, p=2, lam=0.01):
        super().__init__()
        self.p = p
        self.lam = lam
        self.club_estimators = nn.ModuleDict()
        for k, dim in modality_dims.items():
            self.club_estimators[k] = CLUBEstimator(dim, embed_dim)

    def forward(self, modality_inputs, Z_v_debiased_dict):
        mi_terms = {}
        for k in modality_inputs:
            mi_terms[k] = self.club_estimators[k].mi_estimate(
                modality_inputs[k], Z_v_debiased_dict[k]
            )
        return mi_terms

    def adaptive_loss(self, per_modality_losses):
        loss_values = torch.stack([l for l in per_modality_losses.values()])
        p_norm = torch.pow(torch.sum(torch.pow(loss_values, self.p)), 1.0 / self.p)
        return p_norm

    def club_nll(self, modality_inputs, Z_v_debiased_dict):
        total_nll = torch.tensor(0.0, device=modality_inputs[list(modality_inputs.keys())[0]].device)
        for k in modality_inputs:
            total_nll = total_nll + self.club_estimators[k].nll_loss(
                modality_inputs[k], Z_v_debiased_dict[k]
            )
        return total_nll


class InfoNCELoss(nn.Module):
    def __init__(self, tau=0.01):
        super().__init__()
        self.tau = tau

    def forward(self, modality_embs, item_ids=None):
        modality_keys = list(modality_embs.keys())
        M = len(modality_keys)
        if M < 2:
            return torch.tensor(0.0, device=modality_embs[modality_keys[0]].device)
        total_loss = torch.tensor(0.0, device=modality_embs[modality_keys[0]].device)
        n_pairs = 0
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                k1, k2 = modality_keys[i], modality_keys[j]
                z1_raw = modality_embs[k1] if item_ids is None else modality_embs[k1][item_ids]
                z2_raw = modality_embs[k2] if item_ids is None else modality_embs[k2][item_ids]
                z1 = F.normalize(z1_raw, dim=1)
                z2 = F.normalize(z2_raw, dim=1)
                sim_matrix = z1 @ z2.T / self.tau
                labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
                loss = F.cross_entropy(sim_matrix, labels)
                total_loss = total_loss + loss
                n_pairs += 1
        return total_loss / max(n_pairs, 1)


class I2MD4Fair(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims,
                 n_protos=64, eps=0.1, p=2, lam=0.01, tau=0.01,
                 n_layers=2, n_modality_layers=1,
                 lambda1=0.1, lambda2=0.1, lambda3=1e-4):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_modality_layers = n_modality_layers
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lam = lam

        self.user_id_emb = nn.Embedding(n_users, embed_dim)
        self.item_id_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_id_emb.weight)
        nn.init.xavier_uniform_(self.item_id_emb.weight)

        self.intra_mdm = nn.ModuleDict()
        for k in modality_dims:
            self.intra_mdm[k] = IntraMDM(n_items, n_protos, embed_dim, eps)

        self.inter_mdm = InterMDM(modality_dims, embed_dim, p=p, lam=lam)

        self.info_nce = InfoNCELoss(tau=tau)

        self.modality_item_encoders = nn.ModuleDict()
        for k, dim in modality_dims.items():
            encoder = nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            for m in encoder:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.modality_item_encoders[k] = encoder

    def _id_message_passing(self, graph_norm):
        all_embs = torch.cat([self.user_id_emb.weight, self.item_id_emb.weight], dim=0)
        embs_list = [all_embs]
        for _ in range(self.n_layers):
            all_embs = torch.sparse.mm(graph_norm, all_embs) if graph_norm.is_sparse else graph_norm @ all_embs
            embs_list.append(all_embs)
        final = torch.mean(torch.stack(embs_list, dim=1), dim=1)
        return final[:self.n_users], final[self.n_users:]

    def _modality_init_propagation(self, inter_norm_u, inter_norm_v, E_I_init):
        E_I = E_I_init
        for _ in range(self.n_modality_layers):
            if inter_norm_u.is_sparse:
                E_U = torch.sparse.mm(inter_norm_u, E_I)
            else:
                E_U = inter_norm_u @ E_I
            if inter_norm_v.is_sparse:
                E_I = torch.sparse.mm(inter_norm_v, E_U)
            else:
                E_I = inter_norm_v @ E_U
        if self.n_modality_layers == 0:
            Z_I = E_I_init
        else:
            Z_I = (E_I_init + E_I) / (self.n_modality_layers + 1)
        return Z_I

    def _reconstruct_modality_user(self, inter_norm_u, Z_I_debiased):
        if inter_norm_u.is_sparse:
            Z_U = torch.sparse.mm(inter_norm_u, Z_I_debiased)
        else:
            Z_U = inter_norm_u @ Z_I_debiased
        return Z_U

    def forward(self, graph_norm, modality_features, user_ids, item_pos_ids, item_neg_ids,
                interaction_matrix_norm_u=None, interaction_matrix_norm_v=None,
                warmup=False):
        X_U_final, X_V_final = self._id_message_passing(graph_norm)

        Z_I_debiased_dict = {}
        Z_U_dict = {}
        Z_I_dict = {}
        modality_inputs = {}
        batch_item_ids = torch.unique(torch.cat([item_pos_ids, item_neg_ids]))
        for k in modality_features:
            M_k = modality_features[k]
            E_I_k = self.modality_item_encoders[k](M_k)

            if interaction_matrix_norm_u is not None and interaction_matrix_norm_v is not None:
                Z_I_k = self._modality_init_propagation(
                    interaction_matrix_norm_u, interaction_matrix_norm_v, E_I_k
                )
            else:
                Z_I_k = E_I_k

            Z_I_k_debiased, gamma_k = self.intra_mdm[k](Z_I_k)
            Z_I_debiased_dict[k] = Z_I_k_debiased

            if interaction_matrix_norm_u is not None and interaction_matrix_norm_v is not None:
                Z_U_k = self._reconstruct_modality_user(interaction_matrix_norm_u, Z_I_k_debiased)
            else:
                Z_U_k = X_U_final
            Z_U_dict[k] = Z_U_k
            Z_I_dict[k] = Z_I_k_debiased
            modality_inputs[k] = M_k[batch_item_ids]

        mi_terms = self.inter_mdm(
            modality_inputs,
            {k: Z_I_debiased_dict[k][batch_item_ids] for k in Z_I_debiased_dict}
        )

        info_nce_loss = self.info_nce(Z_I_debiased_dict, batch_item_ids)

        hat_X_U = X_U_final
        hat_X_V = X_V_final
        for k in Z_U_dict:
            hat_X_U = torch.cat([hat_X_U, Z_U_dict[k]], dim=1)
            hat_X_V = torch.cat([hat_X_V, Z_I_dict[k]], dim=1)

        u_emb = hat_X_U[user_ids]
        pos_emb = hat_X_V[item_pos_ids]
        neg_emb = hat_X_V[item_neg_ids]

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        per_modality_losses = {}
        for k in modality_features:
            u_emb_k = Z_U_dict[k][user_ids]
            pos_emb_k = Z_I_dict[k][item_pos_ids]
            neg_emb_k = Z_I_dict[k][item_neg_ids]
            pos_scores_k = (u_emb_k * pos_emb_k).sum(dim=1)
            neg_scores_k = (u_emb_k * neg_emb_k).sum(dim=1)
            rec_loss_k = -F.logsigmoid(pos_scores_k - neg_scores_k).mean()
            if warmup:
                total_loss_k = rec_loss_k
            else:
                total_loss_k = rec_loss_k + self.lam * mi_terms[k]
            per_modality_losses[k] = total_loss_k

        adaptive_loss = self.inter_mdm.adaptive_loss(per_modality_losses)

        reg_loss = self._batch_l2_reg(user_ids, item_pos_ids, item_neg_ids)
        total_loss = bpr_loss + self.lambda1 * adaptive_loss + self.lambda2 * info_nce_loss + self.lambda3 * reg_loss

        return total_loss, bpr_loss, adaptive_loss, info_nce_loss

    def _l2_reg(self):
        reg = 0.0
        for param in self.parameters():
            reg += torch.norm(param, 2) ** 2
        return reg / 2

    def _batch_l2_reg(self, user_ids, pos_ids, neg_ids):
        item_ids = torch.unique(torch.cat([pos_ids, neg_ids]))
        reg = (
            self.user_id_emb(user_ids).pow(2).sum() +
            self.item_id_emb(item_ids).pow(2).sum()
        )
        for encoder in self.modality_item_encoders.values():
            for param in encoder.parameters():
                reg = reg + param.pow(2).sum()
        for intra in self.intra_mdm.values():
            reg = reg + intra.hgcn.W.pow(2).sum()
            reg = reg + intra.soft_clustering.prototypes.pow(2).sum()
        return reg / (2.0 * max(user_ids.numel(), 1))

    def club_nll_loss(self, modality_features, item_ids=None):
        Z_I_debiased_dict = {}
        modality_inputs = {}
        for k in modality_features:
            M_k = modality_features[k]
            if item_ids is not None:
                M_k = M_k[item_ids]
            with torch.no_grad():
                E_I_k = self.modality_item_encoders[k](M_k)
            Z_I_debiased_dict[k] = E_I_k.detach()
            modality_inputs[k] = M_k
        return self.inter_mdm.club_nll(modality_inputs, Z_I_debiased_dict)

    def get_user_item_embs(self, graph_norm, modality_features,
                           interaction_matrix_norm_u=None, interaction_matrix_norm_v=None):
        X_U_final, X_V_final = self._id_message_passing(graph_norm)

        Z_I_debiased_dict = {}
        Z_U_dict = {}
        Z_I_dict = {}

        for k in modality_features:
            M_k = modality_features[k]
            E_I_k = self.modality_item_encoders[k](M_k)

            if interaction_matrix_norm_u is not None and interaction_matrix_norm_v is not None:
                Z_I_k = self._modality_init_propagation(
                    interaction_matrix_norm_u, interaction_matrix_norm_v, E_I_k
                )
            else:
                Z_I_k = E_I_k

            Z_I_k_debiased, _ = self.intra_mdm[k](Z_I_k)
            Z_I_debiased_dict[k] = Z_I_k_debiased

            if interaction_matrix_norm_u is not None and interaction_matrix_norm_v is not None:
                Z_U_k = self._reconstruct_modality_user(interaction_matrix_norm_u, Z_I_k_debiased)
            else:
                Z_U_k = X_U_final
            Z_U_dict[k] = Z_U_k
            Z_I_dict[k] = Z_I_k_debiased

        hat_X_U = X_U_final
        hat_X_V = X_V_final
        for k in Z_U_dict:
            hat_X_U = torch.cat([hat_X_U, Z_U_dict[k]], dim=1)
            hat_X_V = torch.cat([hat_X_V, Z_I_dict[k]], dim=1)

        return hat_X_U, hat_X_V
