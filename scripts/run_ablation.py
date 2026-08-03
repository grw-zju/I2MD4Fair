import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DAMRSDataset
from utils.data_utils import BPRDataLoader
from utils.metrics import evaluate_model
from models.i2md4fair import (
    I2MD4Fair, IntraMDM, CLUBEstimator, InfoNCELoss,
    HypergraphConv, SoftPrototypeClustering, SinkhornOT, PairwiseGCN
)


def lloyd_kmeans(X, n_clusters, n_iters=10):
    with torch.no_grad():
        n = X.shape[0]
        k = min(n_clusters, n)
        indices = torch.randperm(n)[:k]
        centroids = X[indices].clone()
        for _ in range(n_iters):
            dists = torch.cdist(X, centroids)
            assignments = dists.argmin(dim=1)
            for t in range(k):
                mask = assignments == t
                if mask.any():
                    centroids[t] = X[mask].mean(dim=0)
        dists = torch.cdist(X, centroids)
        assignments = dists.argmin(dim=1)
        one_hot = torch.zeros(n, k, device=X.device)
        one_hot.scatter_(1, assignments.unsqueeze(1), 1.0)
    return one_hot


class AblationModel(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, modality_dims,
                 use_lightgcn_m=False, use_intra=False, use_soft_proto=False,
                 use_hgcn=False, use_gcn=False, use_inter=False, use_ib=False,
                 use_adaptive=False, use_avg_fusion=False,
                 n_protos=64, eps=0.1, tau=0.01, n_layers=2, n_modality_layers=1,
                 lam=0.01, p_norm=2.0, lambda1=0.1, lambda2=0.1, lambda3=1e-4):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_modality_layers = n_modality_layers
        self.use_lightgcn_m = use_lightgcn_m
        self.use_intra = use_intra
        self.use_soft_proto = use_soft_proto
        self.use_hgcn = use_hgcn
        self.use_gcn = use_gcn
        self.use_inter = use_inter
        self.use_ib = use_ib
        self.use_adaptive = use_adaptive
        self.use_avg_fusion = use_avg_fusion
        self.lam = lam
        self.p_norm = p_norm
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        self.modality_encoders = nn.ModuleDict()
        for k, dim in modality_dims.items():
            encoder = nn.Sequential(
                nn.Linear(dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
            for m in encoder:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self.modality_encoders[k] = encoder

        if use_intra:
            self.intra_mdm = nn.ModuleDict()
            for k in modality_dims:
                self.intra_mdm[k] = IntraMDM(n_items, n_protos, embed_dim, eps)

        if (use_gcn or use_hgcn) and not use_intra and not use_soft_proto:
            self.prop_layers = nn.ModuleDict()
            for k in modality_dims:
                if use_hgcn:
                    self.prop_layers[k] = HypergraphConv(embed_dim)
                else:
                    self.prop_layers[k] = PairwiseGCN(embed_dim)

        if use_soft_proto and not use_intra:
            self.soft_ot = nn.ModuleDict()
            for k in modality_dims:
                self.soft_ot[k] = SinkhornOT(eps=eps, n_iters=3)
            self.prototypes = nn.ParameterDict()
            for k in modality_dims:
                p = torch.empty(n_protos, embed_dim)
                nn.init.xavier_uniform_(p)
                self.prototypes[k] = nn.Parameter(p)
            self.prop_layers = nn.ModuleDict()
            for k in modality_dims:
                if use_hgcn:
                    self.prop_layers[k] = HypergraphConv(embed_dim)
                else:
                    self.prop_layers[k] = PairwiseGCN(embed_dim)

        if use_inter:
            if use_ib:
                self.club_estimators = nn.ModuleDict()
                for k, dim in modality_dims.items():
                    self.club_estimators[k] = CLUBEstimator(dim, embed_dim)

        self.info_nce = InfoNCELoss(tau=tau)

    def _id_message_passing(self, graph_norm):
        all_embs = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
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
            return E_I_init
        return (E_I_init + E_I) / (self.n_modality_layers + 1)

    def _reconstruct_modality_user(self, inter_norm_u, Z_I_debiased):
        if inter_norm_u.is_sparse:
            return torch.sparse.mm(inter_norm_u, Z_I_debiased)
        return inter_norm_u @ Z_I_debiased

    def forward(self, graph_norm, modality_features, user_ids, pos_ids, neg_ids,
                interaction_matrix_norm_u=None, interaction_matrix_norm_v=None,
                warmup=False):
        user_embs, item_embs = self._id_message_passing(graph_norm)

        Z_I_dict = {}
        Z_U_dict = {}
        Z_I_debiased_dict = {}
        batch_item_ids = torch.unique(torch.cat([pos_ids, neg_ids]))

        for k in modality_features:
            E_I_k = self.modality_encoders[k](modality_features[k])

            if interaction_matrix_norm_u is not None and interaction_matrix_norm_v is not None:
                Z_I_k = self._modality_init_propagation(
                    interaction_matrix_norm_u, interaction_matrix_norm_v, E_I_k)
            else:
                Z_I_k = E_I_k

            if self.use_intra:
                Z_I_k_debiased, _ = self.intra_mdm[k](Z_I_k)
            elif self.use_hgcn and not self.use_soft_proto:
                one_hot = lloyd_kmeans(Z_I_k, self.prototypes_abl[k].shape[0] if hasattr(self, 'prototypes_abl') else 64)
                incidence = Z_I_k.shape[0] * one_hot
                Z_I_k_debiased = self.prop_layers[k](Z_I_k, incidence)
            elif self.use_soft_proto:
                Z_norm = F.normalize(Z_I_k, dim=1)
                P_norm = F.normalize(self.prototypes[k], dim=1)
                gamma = self.soft_ot[k](1 - Z_norm @ P_norm.T)
                incidence = Z_I_k.shape[0] * gamma
                Z_I_k_debiased = self.prop_layers[k](Z_I_k, incidence)
            elif self.use_gcn:
                one_hot = lloyd_kmeans(Z_I_k, 64)
                incidence = Z_I_k.shape[0] * one_hot
                Z_I_k_debiased = self.prop_layers[k](Z_I_k, incidence)
            else:
                Z_I_k_debiased = Z_I_k

            Z_I_debiased_dict[k] = Z_I_k_debiased
            Z_I_dict[k] = Z_I_k_debiased

            if interaction_matrix_norm_u is not None and interaction_matrix_norm_v is not None:
                Z_U_k = self._reconstruct_modality_user(interaction_matrix_norm_u, Z_I_k_debiased)
            else:
                Z_U_k = user_embs
            Z_U_dict[k] = Z_U_k

        hat_X_U = user_embs
        hat_X_V = item_embs
        for k in Z_U_dict:
            hat_X_U = torch.cat([hat_X_U, Z_U_dict[k]], dim=1)
            hat_X_V = torch.cat([hat_X_V, Z_I_dict[k]], dim=1)

        u = hat_X_U[user_ids]
        pos = hat_X_V[pos_ids]
        neg = hat_X_V[neg_ids]
        pos_scores = (u * pos).sum(dim=1)
        neg_scores = (u * neg).sum(dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        per_modality_losses = {}
        for k in modality_features:
            u_k = Z_U_dict[k][user_ids]
            pos_k = Z_I_dict[k][pos_ids]
            neg_k = Z_I_dict[k][neg_ids]
            pos_s = (u_k * pos_k).sum(dim=1)
            neg_s = (u_k * neg_k).sum(dim=1)
            rec_loss_k = -F.logsigmoid(pos_s - neg_s).mean()

            if self.use_ib:
                M_k = modality_features[k][batch_item_ids]
                Z_k = Z_I_debiased_dict[k][batch_item_ids]
                mi_est = self.club_estimators[k].mi_upper_bound(M_k, Z_k)
                if warmup:
                    total_k = rec_loss_k
                else:
                    total_k = rec_loss_k + self.lam * mi_est
            else:
                total_k = rec_loss_k
            per_modality_losses[k] = total_k

        if self.use_adaptive:
            loss_values = torch.stack([l for l in per_modality_losses.values()])
            adaptive_loss = torch.pow(torch.sum(torch.pow(loss_values, self.p_norm)), 1.0 / self.p_norm)
        elif self.use_avg_fusion:
            adaptive_loss = sum(per_modality_losses.values()) / len(per_modality_losses)
        elif self.use_inter:
            adaptive_loss = sum(per_modality_losses.values())
        else:
            adaptive_loss = sum(per_modality_losses.values())

        info_nce_loss = self.info_nce(Z_I_debiased_dict, batch_item_ids)

        l2_reg = sum(torch.norm(p, 2) ** 2 for p in self.parameters()) / 2

        total_loss = bpr_loss + self.lambda1 * adaptive_loss + self.lambda2 * info_nce_loss + self.lambda3 * l2_reg
        return total_loss, bpr_loss, adaptive_loss, info_nce_loss

    def club_nll_loss(self, modality_features, item_ids=None):
        if not self.use_ib:
            return torch.tensor(0.0)
        total_nll = torch.tensor(0.0, device=next(self.parameters()).device)
        for k in self.club_estimators:
            M_k = modality_features[k]
            if item_ids is not None:
                M_k = M_k[item_ids]
            Z_k = self.modality_encoders[k](M_k)
            if self.use_intra:
                Z_k, _ = self.intra_mdm[k](Z_k)
            Z_k = Z_k.detach()
            total_nll = total_nll + self.club_estimators[k].nll_loss(M_k, Z_k)
        return total_nll

    def get_user_item_embs(self, graph_norm, modality_features,
                           interaction_matrix_norm_u=None, interaction_matrix_norm_v=None):
        user_embs, item_embs = self._id_message_passing(graph_norm)

        Z_I_dict = {}
        Z_U_dict = {}

        for k in modality_features:
            E_I_k = self.modality_encoders[k](modality_features[k])

            if interaction_matrix_norm_u is not None and interaction_matrix_norm_v is not None:
                Z_I_k = self._modality_init_propagation(
                    interaction_matrix_norm_u, interaction_matrix_norm_v, E_I_k)
            else:
                Z_I_k = E_I_k

            if self.use_intra:
                Z_I_k_debiased, _ = self.intra_mdm[k](Z_I_k)
            elif self.use_hgcn and not self.use_soft_proto:
                one_hot = lloyd_kmeans(Z_I_k, 64)
                incidence = Z_I_k.shape[0] * one_hot
                Z_I_k_debiased = self.prop_layers[k](Z_I_k, incidence)
            elif self.use_soft_proto:
                Z_norm = F.normalize(Z_I_k, dim=1)
                P_norm = F.normalize(self.prototypes[k], dim=1)
                gamma = self.soft_ot[k](1 - Z_norm @ P_norm.T)
                incidence = Z_I_k.shape[0] * gamma
                Z_I_k_debiased = self.prop_layers[k](Z_I_k, incidence)
            elif self.use_gcn:
                one_hot = lloyd_kmeans(Z_I_k, 64)
                incidence = Z_I_k.shape[0] * one_hot
                Z_I_k_debiased = self.prop_layers[k](Z_I_k, incidence)
            else:
                Z_I_k_debiased = Z_I_k

            Z_I_dict[k] = Z_I_k_debiased

            if interaction_matrix_norm_u is not None and interaction_matrix_norm_v is not None:
                Z_U_k = self._reconstruct_modality_user(interaction_matrix_norm_u, Z_I_k_debiased)
            else:
                Z_U_k = user_embs
            Z_U_dict[k] = Z_U_k

        hat_X_U = user_embs
        hat_X_V = item_embs
        for k in Z_U_dict:
            hat_X_U = torch.cat([hat_X_U, Z_U_dict[k]], dim=1)
            hat_X_V = torch.cat([hat_X_V, Z_I_dict[k]], dim=1)
        return hat_X_U, hat_X_V


def ablation_experiment(args, device):
    dataset = DAMRSDataset(args.dataset, args.data_dir, args.embed_dim)
    modality_dims = dataset.get_modality_features_dim()

    configs = [
        ('LightGCN+M', 'lightgcn_m'),
        ('+ k-means, GCN', 'kmeans_gcn'),
        ('+ k-means, HGCN', 'kmeans_hgcn'),
        ('+ soft prototype, GCN', 'soft_proto_gcn'),
        ('+ Intra', 'intra_only'),
        ('+ IB', 'ib_only'),
        ('+ IB, Average fusion', 'ib_avg_fusion'),
        ('+ Adaptive p-norm', 'adaptive_pnorm'),
        ('+ Inter', 'inter_only'),
        ('I2MD4Fair', 'full'),
    ]

    all_results = {}
    for name, config in configs:
        print(f"\n=== Ablation: {name} ===")
        results = run_ablation_config(config, dataset, modality_dims, args, device, n_runs=args.n_runs)
        all_results[name] = results

    print_ablation_table(all_results)
    return all_results


def run_ablation_config(config, dataset, modality_dims, args, device, n_runs=5):
    results_all = defaultdict(list)

    for run in range(n_runs):
        model = build_ablation_model(config, dataset, modality_dims, args, device)

        club_params = set()
        if hasattr(model, 'club_estimators'):
            for estimator in model.club_estimators.values():
                club_params.update(estimator.parameters())
        main_params = [p for p in model.parameters() if p not in club_params]
        main_optimizer = torch.optim.Adam(main_params, lr=args.lr)
        club_optimizer = torch.optim.Adam(list(club_params), lr=args.lr) if club_params else None

        graph_norm = dataset.get_norm_graph().to(device)
        norm_matrices = dataset.get_modality_norm_matrices()
        inter_norm_u = norm_matrices['inter_norm_u'].to(device)
        inter_norm_v = norm_matrices['inter_norm_v'].to(device)
        modality_features = dataset.get_modality_features()
        for k in modality_features:
            modality_features[k] = modality_features[k].to(device)

        data_loader = BPRDataLoader(
            dataset.train_data, dataset.n_users, dataset.n_items,
            dataset.train_user_item_dict, batch_size=args.batch_size,
            user_item_dict=dataset.user_item_dict
        )

        best_r10 = -1.0
        best_metrics = None
        best_state = None
        patience = 0

        for epoch in range(1, args.max_epochs + 1):
            model.train()
            is_warmup = epoch <= args.warmup_epochs
            for _ in range(len(data_loader)):
                user_ids, pos_ids, neg_ids = data_loader.get_batch()
                user_ids = user_ids.to(device)
                pos_ids = pos_ids.to(device)
                neg_ids = neg_ids.to(device)
                batch_item_ids = torch.unique(torch.cat([pos_ids, neg_ids]))

                if club_optimizer is not None:
                    club_optimizer.zero_grad()
                    club_nll = model.club_nll_loss(modality_features, batch_item_ids)
                    club_nll.backward()
                    torch.nn.utils.clip_grad_norm_(list(club_params), max_norm=5.0)
                    club_optimizer.step()

                main_optimizer.zero_grad()
                if config == 'full':
                    loss, _, _, _ = model(
                        graph_norm, modality_features, user_ids, pos_ids, neg_ids,
                        interaction_matrix_norm_u=inter_norm_u,
                        interaction_matrix_norm_v=inter_norm_v,
                        warmup=is_warmup)
                else:
                    loss, _, _, _ = model(
                        graph_norm, modality_features, user_ids, pos_ids, neg_ids,
                        interaction_matrix_norm_u=inter_norm_u,
                        interaction_matrix_norm_v=inter_norm_v,
                        warmup=is_warmup)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(main_params, max_norm=5.0)
                main_optimizer.step()

            if epoch % args.eval_interval == 0:
                metrics = evaluate_model(model, dataset, K_list=[10, 20], device=device, mode='val')
                r10 = metrics['Recall'][10]
                if r10 > best_r10:
                    best_r10 = r10
                    best_metrics = metrics
                    best_state = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                if patience >= args.patience:
                    break

        if best_metrics:
            if best_state is not None:
                model.load_state_dict(best_state)
            test_metrics = evaluate_model(model, dataset, K_list=[10, 20], device=device, mode='test')
            for m in test_metrics:
                for K in test_metrics[m]:
                    results_all[(m, K)].append(test_metrics[m][K])

    avg = {}
    for key in results_all:
        avg[key] = np.mean(results_all[key])
    return avg


def build_ablation_model(config, dataset, modality_dims, args, device):
    n_users = dataset.n_users
    n_items = dataset.n_items
    embed_dim = args.embed_dim

    if config == 'full':
        return I2MD4Fair(
            n_users, n_items, embed_dim, modality_dims,
            n_protos=args.n_protos, eps=args.eps, p=args.p_norm,
            lam=args.lam, tau=args.tau, n_layers=args.n_layers,
            n_modality_layers=args.n_modality_layers,
            lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3
        ).to(device)

    config_map = {
        'lightgcn_m': dict(use_lightgcn_m=True),
        'kmeans_gcn': dict(use_gcn=True),
        'kmeans_hgcn': dict(use_hgcn=True),
        'soft_proto_gcn': dict(use_soft_proto=True, use_gcn=True),
        'intra_only': dict(use_intra=True),
        'ib_only': dict(use_inter=True, use_ib=True),
        'ib_avg_fusion': dict(use_inter=True, use_ib=True, use_avg_fusion=True),
        'adaptive_pnorm': dict(use_inter=True, use_adaptive=True),
        'inter_only': dict(use_inter=True, use_ib=True, use_adaptive=True),
    }

    cfg = config_map[config]
    return AblationModel(n_users, n_items, embed_dim, modality_dims,
                        n_protos=args.n_protos, eps=args.eps, tau=args.tau,
                        n_layers=args.n_layers, n_modality_layers=args.n_modality_layers,
                        lam=args.lam, p_norm=args.p_norm,
                        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
                        **cfg).to(device)


def print_ablation_table(all_results):
    print("\n" + "=" * 160)
    print(f"{'Model':<30} {'N@10':>8} {'N@20':>8} {'R@10':>8} {'R@20':>8} "
          f"{'HR@10':>8} {'HR@20':>8} "
          f"{'G@10':>8} {'G@20':>8} {'E@10':>8} {'E@20':>8} {'C@10':>8} {'C@20':>8} "
          f"{'REG@10':>8} {'REG@20':>8}")
    print("=" * 160)
    for name in all_results:
        r = all_results[name]
        print(f"{name:<30} "
              f"{r.get(('NDCG', 10), 0):>8.4f} {r.get(('NDCG', 20), 0):>8.4f} "
              f"{r.get(('Recall', 10), 0):>8.4f} {r.get(('Recall', 20), 0):>8.4f} "
              f"{r.get(('HR', 10), 0):>8.4f} {r.get(('HR', 20), 0):>8.4f} "
              f"{r.get(('Gini', 10), 0):>8.4f} {r.get(('Gini', 20), 0):>8.4f} "
              f"{r.get(('Entropy', 10), 0):>8.4f} {r.get(('Entropy', 20), 0):>8.4f} "
              f"{r.get(('Coverage', 10), 0):>8.4f} {r.get(('Coverage', 20), 0):>8.4f} "
              f"{r.get(('REG', 10), 0):>8.4f} {r.get(('REG', 20), 0):>8.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='baby', choices=['baby', 'clothing', 'mind', 'demo'])
    parser.add_argument('--data_dir', type=str, default='data/damrs/')
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_modality_layers', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--n_protos', type=int, default=64)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--p_norm', type=float, default=2.0)
    parser.add_argument('--lam', '--lambda_ib', type=float, default=0.01, dest='lam')
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--lambda1', '--lambda_amb', type=float, default=0.1, dest='lambda1')
    parser.add_argument('--lambda2', '--lambda_cl', type=float, default=0.1, dest='lambda2')
    parser.add_argument('--lambda3', '--lambda_reg', type=float, default=1e-4, dest='lambda3')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ablation_experiment(args, device)
