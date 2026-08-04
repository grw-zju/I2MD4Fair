import os
import sys
import argparse
import torch
import numpy as np
import copy
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import DAMRSDataset
from utils.data_utils import BPRDataLoader
from utils.metrics import evaluate_model
from models.i2md4fair import I2MD4Fair


def hyperparameter_experiment(args, device):
    dataset = DAMRSDataset(args.dataset, args.data_dir, args.embed_dim)
    modality_dims = dataset.get_modality_features_dim()

    lambda1_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    lambda2_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    lam_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    p_values = [1, 2, 4, 8]
    T_values = [16, 32, 64, 128, 256]

    results = {}

    print("\n=== Lambda1 (Adaptive Modality-Balanced Loss) ===")
    results['lambda1'] = {}
    orig_lambda1 = args.lambda1
    for lam1 in lambda1_values:
        args.lambda1 = lam1
        avg = train_single(args, dataset, modality_dims, device, n_runs=args.n_runs)
        results['lambda1'][lam1] = avg
        print(f"lambda1={lam1}: N@10={avg.get(('NDCG',10),0):.4f}, R@10={avg.get(('Recall',10),0):.4f}, "
              f"C@10={avg.get(('Coverage',10),0):.4f}")
    args.lambda1 = orig_lambda1

    print("\n=== Lambda2 (Cross-modality Alignment Loss) ===")
    results['lambda2'] = {}
    orig_lambda2 = args.lambda2
    for lam2 in lambda2_values:
        args.lambda2 = lam2
        avg = train_single(args, dataset, modality_dims, device, n_runs=args.n_runs)
        results['lambda2'][lam2] = avg
        print(f"lambda2={lam2}: N@10={avg.get(('NDCG',10),0):.4f}, R@10={avg.get(('Recall',10),0):.4f}, "
              f"C@10={avg.get(('Coverage',10),0):.4f}")
    args.lambda2 = orig_lambda2

    print("\n=== Lambda (Modality Info Regularization) ===")
    results['lam'] = {}
    orig_lam = args.lam
    for lam_val in lam_values:
        args.lam = lam_val
        avg = train_single(args, dataset, modality_dims, device, n_runs=args.n_runs)
        results['lam'][lam_val] = avg
        print(f"lambda={lam_val}: N@10={avg.get(('NDCG',10),0):.4f}, R@10={avg.get(('Recall',10),0):.4f}, "
              f"C@10={avg.get(('Coverage',10),0):.4f}")
    args.lam = orig_lam

    print("\n=== p-norm (Adaptive Fusion Order) ===")
    results['p'] = {}
    orig_p = args.p_norm
    for p_val in p_values:
        args.p_norm = p_val
        avg = train_single(args, dataset, modality_dims, device, n_runs=args.n_runs)
        results['p'][p_val] = avg
        print(f"p={p_val}: N@10={avg.get(('NDCG',10),0):.4f}, R@10={avg.get(('Recall',10),0):.4f}, "
              f"C@10={avg.get(('Coverage',10),0):.4f}")
    args.p_norm = orig_p

    print("\n=== T (Number of Prototypes) ===")
    results['T'] = {}
    orig_T = args.n_protos
    for T_val in T_values:
        args.n_protos = T_val
        avg = train_single(args, dataset, modality_dims, device, n_runs=args.n_runs)
        results['T'][T_val] = avg
        print(f"T={T_val}: N@10={avg.get(('NDCG',10),0):.4f}, R@10={avg.get(('Recall',10),0):.4f}, "
              f"C@10={avg.get(('Coverage',10),0):.4f}")
    args.n_protos = orig_T

    save_results(results, args.dataset)
    return results


def train_single(args, dataset, modality_dims, device, n_runs=3):
    results_all = defaultdict(list)

    for run in range(n_runs):
        model = I2MD4Fair(
            dataset.n_users, dataset.n_items, args.embed_dim, modality_dims,
            n_protos=args.n_protos, eps=args.eps, p=args.p_norm,
            lam=args.lam, tau=args.tau, n_layers=args.n_layers,
            n_modality_layers=args.n_modality_layers,
            lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3
        ).to(device)

        club_params = set()
        for estimator in model.inter_mdm.club_estimators.values():
            club_params.update(estimator.parameters())
        main_params = [p for p in model.parameters() if p not in club_params]
        optimizer = torch.optim.Adam(main_params, lr=args.lr)
        club_optimizer = torch.optim.Adam(list(club_params), lr=args.lr)
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

                Z_I_full_dict = model._compute_modality_reprs(
                    modality_features, inter_norm_u, inter_norm_v, detach=False)
                club_Z_I_full_dict = {k: v.detach() for k, v in Z_I_full_dict.items()}

                club_optimizer.zero_grad()
                club_nll = model.club_nll_loss(
                    modality_features, graph_norm=graph_norm,
                    interaction_matrix_norm_u=inter_norm_u,
                    interaction_matrix_norm_v=inter_norm_v,
                    item_ids=batch_item_ids, Z_I_full_dict=club_Z_I_full_dict)
                club_nll.backward()
                torch.nn.utils.clip_grad_norm_(list(club_params), max_norm=5.0)
                club_optimizer.step()

                optimizer.zero_grad()
                loss, _, _, _ = model(
                    graph_norm, modality_features, user_ids, pos_ids, neg_ids,
                    interaction_matrix_norm_u=inter_norm_u,
                    interaction_matrix_norm_v=inter_norm_v,
                    warmup=is_warmup, Z_I_full_dict=Z_I_full_dict
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(main_params, max_norm=5.0)
                optimizer.step()

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


def save_results(results, dataset_name):
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    import json
    serializable = {}
    for param_name in results:
        serializable[param_name] = {}
        for val in results[param_name]:
            key_str = str(val)
            serializable[param_name][key_str] = {
                str(k): v for k, v in results[param_name][val].items()
            }
    with open(os.path.join(output_dir, f'{dataset_name}_hyperparam.json'), 'w') as f:
        json.dump(serializable, f, indent=2)


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
    hyperparameter_experiment(args, device)
