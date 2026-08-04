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
from models.i2md4fair import IntraMDM, CLUBEstimator, InfoNCELoss


class _PatchedEncoder(nn.Module):
    def __init__(self, original_encoder, intra_mdm=None):
        super().__init__()
        self.original_encoder = original_encoder
        self.intra_mdm = intra_mdm

    def forward(self, x):
        z = self.original_encoder(x)
        if self.intra_mdm is not None:
            z, _ = self.intra_mdm(z)
        return z


class BackboneWithDebias(nn.Module):
    def __init__(self, base_model, modality_dims, embed_dim,
                 use_intra=False, use_inter=False,
                 n_protos=64, eps=0.1, p=2, lam=0.01, tau=0.01,
                 lambda1=0.1, lambda2=0.1):
        super().__init__()
        self.base_model = base_model
        self.n_users = base_model.n_users
        self.n_items = base_model.n_items
        self.embed_dim = embed_dim
        self.use_intra = use_intra
        self.use_inter = use_inter
        self.lam = lam
        self.p = p
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        if use_intra:
            self.intra_mdm = nn.ModuleDict()
            for k in modality_dims:
                self.intra_mdm[k] = IntraMDM(self.n_items, n_protos, embed_dim, eps)
            for k in modality_dims:
                if hasattr(base_model, 'modality_encoders') and k in base_model.modality_encoders:
                    base_model.modality_encoders[k] = _PatchedEncoder(
                        base_model.modality_encoders[k], self.intra_mdm[k])
                elif hasattr(base_model, 'modality_item_encoders') and k in base_model.modality_item_encoders:
                    base_model.modality_item_encoders[k] = _PatchedEncoder(
                        base_model.modality_item_encoders[k], self.intra_mdm[k])
                elif hasattr(base_model, 'modality_trans') and k in base_model.modality_trans:
                    base_model.modality_trans[k] = _PatchedEncoder(
                        base_model.modality_trans[k], self.intra_mdm[k])

        if use_inter:
            self.club_estimators = nn.ModuleDict()
            for k, dim in modality_dims.items():
                self.club_estimators[k] = CLUBEstimator(dim, embed_dim)

        self.info_nce = InfoNCELoss(tau=tau)

    def _get_raw_modality_item_embs(self, modality_features):
        result = {}
        for k in modality_features:
            if hasattr(self.base_model, 'modality_encoders') and k in self.base_model.modality_encoders:
                enc = self.base_model.modality_encoders[k]
                if isinstance(enc, _PatchedEncoder):
                    result[k] = enc.original_encoder(modality_features[k])
                else:
                    result[k] = enc(modality_features[k])
            elif hasattr(self.base_model, 'modality_item_encoders') and k in self.base_model.modality_item_encoders:
                enc = self.base_model.modality_item_encoders[k]
                if isinstance(enc, _PatchedEncoder):
                    result[k] = enc.original_encoder(modality_features[k])
                else:
                    result[k] = enc(modality_features[k])
            elif hasattr(self.base_model, 'modality_trans') and k in self.base_model.modality_trans:
                enc = self.base_model.modality_trans[k]
                if isinstance(enc, _PatchedEncoder):
                    result[k] = enc.original_encoder(modality_features[k])
                else:
                    result[k] = enc(modality_features[k])
            else:
                feat = modality_features[k]
                if feat.shape[1] >= self.embed_dim:
                    result[k] = feat[:, :self.embed_dim]
                else:
                    result[k] = F.pad(feat, (0, self.embed_dim - feat.shape[1]))
        return result

    def compute_loss(self, user_ids, pos_ids, neg_ids, graph_norm, modality_features,
                     inter_norm_u=None, inter_norm_v=None, warmup=False):
        base_loss = self.base_model.compute_loss(user_ids, pos_ids, neg_ids, graph_norm, modality_features)

        if not self.use_inter:
            return base_loss

        raw_item_embs = self._get_raw_modality_item_embs(modality_features)
        Z_I_debiased_dict = {}
        for k in raw_item_embs:
            Z_I_k = raw_item_embs[k]
            if self.use_intra:
                Z_I_k_debiased, _ = self.intra_mdm[k](Z_I_k)
            else:
                Z_I_k_debiased = Z_I_k
            Z_I_debiased_dict[k] = Z_I_k_debiased

        batch_item_ids = torch.unique(torch.cat([pos_ids, neg_ids]))

        if not warmup:
            mi_terms = {}
            for k in modality_features:
                M_k = modality_features[k][batch_item_ids]
                Z_k = Z_I_debiased_dict[k][batch_item_ids]
                mi_terms[k] = self.club_estimators[k].mi_upper_bound(M_k, Z_k)

            per_mod_losses = {}
            for k in modality_features:
                u_emb_k = self._get_modality_user_emb(k, user_ids)
                pos_k = Z_I_debiased_dict[k][pos_ids]
                neg_k = Z_I_debiased_dict[k][neg_ids]
                pos_s = (u_emb_k * pos_k).sum(dim=1)
                neg_s = (u_emb_k * neg_k).sum(dim=1)
                rec_k = -F.logsigmoid(pos_s - neg_s).mean()
                per_mod_losses[k] = rec_k + self.lam * mi_terms[k]

            loss_values = torch.stack([l for l in per_mod_losses.values()]).clamp_min(1e-12)
            adaptive_loss = torch.pow(torch.sum(torch.pow(loss_values, self.p)), 1.0 / self.p)
            base_loss = base_loss + self.lambda1 * adaptive_loss

        info_loss = self.info_nce(Z_I_debiased_dict, batch_item_ids)
        base_loss = base_loss + self.lambda2 * info_loss

        return base_loss

    def _get_modality_user_emb(self, k, user_ids):
        if hasattr(self.base_model, 'modality_user_embs') and k in self.base_model.modality_user_embs:
            return self.base_model.modality_user_embs[k](user_ids)
        if hasattr(self.base_model, 'user_emb'):
            return self.base_model.user_emb(user_ids)
        if hasattr(self.base_model, 'user_embedding'):
            return self.base_model.user_embedding(user_ids)
        return torch.zeros(len(user_ids), self.embed_dim, device=user_ids.device)

    def club_nll_loss(self, modality_features, item_ids=None):
        if not self.use_inter:
            return torch.tensor(0.0)
        total_nll = torch.tensor(0.0, device=next(self.parameters()).device)
        for k in self.club_estimators:
            M_k = modality_features[k]
            if item_ids is not None:
                M_k = M_k[item_ids]
            Z_k = self._get_raw_modality_item_embs({k: M_k})[k]
            if self.use_intra:
                Z_k, _ = self.intra_mdm[k](Z_k)
            Z_k = Z_k.detach()
            total_nll = total_nll + self.club_estimators[k].nll_loss(M_k, Z_k)
        return total_nll

    def get_embs(self, graph_norm, modality_features):
        return self.base_model.get_embs(graph_norm, modality_features)

    def get_user_item_embs(self, graph_norm, modality_features,
                           interaction_matrix_norm_u=None, interaction_matrix_norm_v=None):
        return self.base_model.get_embs(graph_norm, modality_features)


def build_backbone_model(backbone_name, dataset, args, device):
    from baseline import MMSSL, DiffMM, LGMRec, MENTOR
    n_users = dataset.n_users
    n_items = dataset.n_items
    embed_dim = args.embed_dim
    modality_dims = dataset.get_modality_features_dim()

    if backbone_name == 'MMSSL':
        model = MMSSL(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif backbone_name == 'DiffMM':
        model = DiffMM(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif backbone_name == 'LGMRec':
        model = LGMRec(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif backbone_name == 'MENTOR':
        model = MENTOR(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    target = model.base_model if hasattr(model, 'base_model') else model
    if hasattr(target, 'set_precomputed_adj'):
        target.set_precomputed_adj(dataset.get_adj_matrices())
    if hasattr(target, 'set_train_interactions'):
        target.set_train_interactions(dataset.train_data, dataset.n_users, dataset.n_items)

    return model.to(device)


def run_transfer_experiment(args, device):
    dataset = DAMRSDataset(args.dataset, args.data_dir, args.embed_dim)
    modality_dims = dataset.get_modality_features_dim()

    backbones = ['MMSSL', 'DiffMM', 'LGMRec', 'MENTOR']
    variants = [
        ('Backbone', False, False),
        ('+ Intra-MDM', True, False),
        ('+ Inter-MDM', False, True),
        ('+ Intra-MDM + Inter-MDM', True, True),
    ]

    all_results = {}

    for backbone_name in backbones:
        print(f"\n{'='*80}")
        print(f"Backbone: {backbone_name}")
        print(f"{'='*80}")

        for variant_name, use_intra, use_inter in variants:
            config_key = f"{backbone_name} | {variant_name}"
            print(f"\n--- {config_key} ---")

            results = run_single_config(
                backbone_name, use_intra, use_inter,
                dataset, modality_dims, args, device, n_runs=args.n_runs
            )
            all_results[config_key] = results

    print_transfer_table(all_results, backbones)
    return all_results


def run_single_config(backbone_name, use_intra, use_inter,
                      dataset, modality_dims, args, device, n_runs=5):
    results_all = defaultdict(list)

    for run in range(n_runs):
        torch.manual_seed(run)
        np.random.seed(run)

        base_model = build_backbone_model(backbone_name, dataset, args, device)

        if use_intra or use_inter:
            model = BackboneWithDebias(
                base_model, modality_dims, args.embed_dim,
                use_intra=use_intra, use_inter=use_inter,
                n_protos=args.n_protos, eps=args.eps, p=args.p_norm,
                lam=args.lam, tau=args.tau,
                lambda1=args.lambda1, lambda2=args.lambda2
            ).to(device)
        else:
            model = base_model

        club_params = set()
        if use_inter:
            for estimator in model.club_estimators.values():
                club_params.update(estimator.parameters())
        main_params = [p for p in model.parameters() if p not in club_params]
        optimizer = torch.optim.Adam(main_params, lr=args.lr)
        club_optimizer = torch.optim.Adam(list(club_params), lr=args.lr) if club_params else None

        graph_norm = dataset.get_norm_graph().to(device)
        modality_features = dataset.get_modality_features()
        for k in modality_features:
            modality_features[k] = modality_features[k].to(device)

        norm_matrices = dataset.get_modality_norm_matrices()
        inter_norm_u = norm_matrices['inter_norm_u'].to(device)
        inter_norm_v = norm_matrices['inter_norm_v'].to(device)

        data_loader = BPRDataLoader(
            dataset.train_data, dataset.n_users, dataset.n_items,
            dataset.train_user_item_dict, batch_size=args.batch_size,
            user_item_dict=dataset.user_item_dict
        )

        best_r10 = -1.0
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

                if use_inter and club_optimizer is not None:
                    club_optimizer.zero_grad()
                    club_nll = model.club_nll_loss(modality_features, batch_item_ids)
                    club_nll.backward()
                    torch.nn.utils.clip_grad_norm_(list(club_params), max_norm=5.0)
                    club_optimizer.step()

                optimizer.zero_grad()

                if use_intra or use_inter:
                    loss = model.compute_loss(
                        user_ids, pos_ids, neg_ids, graph_norm, modality_features,
                        inter_norm_u=inter_norm_u, inter_norm_v=inter_norm_v,
                        warmup=is_warmup
                    )
                else:
                    loss = model.compute_loss(user_ids, pos_ids, neg_ids, graph_norm, modality_features)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(main_params, max_norm=5.0)
                optimizer.step()

            if epoch % args.eval_interval == 0:
                metrics = evaluate_model(model, dataset, K_list=[10, 20], device=device, mode='val')
                r10 = metrics['Recall'][10]
                if r10 > best_r10:
                    best_r10 = r10
                    best_state = copy.deepcopy(model.state_dict())
                    patience = 0
                else:
                    patience += 1
                if patience >= args.patience:
                    break

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


def print_transfer_table(all_results, backbones):
    print("\n" + "=" * 160)
    print(f"{'Backbone Variant':<30} {'N@10':>8} {'N@20':>8} {'R@10':>8} {'R@20':>8} "
          f"{'HR@10':>8} {'HR@20':>8} "
          f"{'G@10':>8} {'G@20':>8} {'E@10':>8} {'E@20':>8} {'C@10':>8} {'C@20':>8} "
          f"{'REG@10':>8} {'REG@20':>8}")
    print("=" * 160)
    for backbone_name in backbones:
        for variant_name in ['Backbone', '+ Intra-MDM', '+ Inter-MDM', '+ Intra-MDM + Inter-MDM']:
            key = f"{backbone_name} | {variant_name}"
            r = all_results.get(key, {})
            label = f"{backbone_name} {variant_name}" if variant_name != 'Backbone' else backbone_name
            print(f"{label:<30} "
                  f"{r.get(('NDCG', 10), 0):>8.4f} {r.get(('NDCG', 20), 0):>8.4f} "
                  f"{r.get(('Recall', 10), 0):>8.4f} {r.get(('Recall', 20), 0):>8.4f} "
                  f"{r.get(('HR', 10), 0):>8.4f} {r.get(('HR', 20), 0):>8.4f} "
                  f"{r.get(('Gini', 10), 0):>8.4f} {r.get(('Gini', 20), 0):>8.4f} "
                  f"{r.get(('Entropy', 10), 0):>8.4f} {r.get(('Entropy', 20), 0):>8.4f} "
                  f"{r.get(('Coverage', 10), 0):>8.4f} {r.get(('Coverage', 20), 0):>8.4f} "
                  f"{r.get(('REG', 10), 0):>8.4f} {r.get(('REG', 20), 0):>8.4f}")
        print("-" * 160)


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
    parser.add_argument('--lam', type=float, default=0.01)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--lambda3', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    run_transfer_experiment(args, device)
