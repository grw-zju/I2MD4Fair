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
from baseline import (
    LightGCN, VBPR, MMGCN, GRCN, LATTICE, FREEDOM,
    LGMRec, BM3, SLMRec, MMSSL, DiffMM, MENTOR,
    DMRL, CLUSSL, ModalityDebiasingWrapper, DPRWrapper, FairDualWrapper
)
from scripts.run_transfer import BackboneWithDebias

MODEL_REGISTRY = {
    'I2MD4Fair': I2MD4Fair,
    'LightGCN': LightGCN,
    'VBPR': VBPR,
    'MMGCN': MMGCN,
    'GRCN': GRCN,
    'LATTICE': LATTICE,
    'FREEDOM': FREEDOM,
    'LGMRec': LGMRec,
    'BM3': BM3,
    'SLMRec': SLMRec,
    'MMSSL': MMSSL,
    'DiffMM': DiffMM,
    'MENTOR': MENTOR,
    'DMRL': DMRL,
    'CLUSSL': CLUSSL,
    'MMSSL+MD': MMSSL,
    'DiffMM+MD': DiffMM,
    'LGMRec+MD': LGMRec,
    'MENTOR+MD': MENTOR,
    'MMSSL+DPR': MMSSL,
    'DiffMM+DPR': DiffMM,
    'LGMRec+DPR': LGMRec,
    'MENTOR+DPR': MENTOR,
    'MMSSL+FairDual': MMSSL,
    'DiffMM+FairDual': DiffMM,
    'LGMRec+FairDual': LGMRec,
    'MENTOR+FairDual': MENTOR,
    'MMSSL+Intra': MMSSL,
    'DiffMM+Intra': DiffMM,
    'LGMRec+Intra': LGMRec,
    'MENTOR+Intra': MENTOR,
    'MMSSL+Inter': MMSSL,
    'DiffMM+Inter': DiffMM,
    'LGMRec+Inter': LGMRec,
    'MENTOR+Inter': MENTOR,
    'MMSSL+Intra+Inter': MMSSL,
    'DiffMM+Intra+Inter': DiffMM,
    'LGMRec+Intra+Inter': LGMRec,
    'MENTOR+Intra+Inter': MENTOR,
}

GRAPH_ONLY_MODELS = {'LightGCN'}
MODALITY_ONLY_MODELS = {'VBPR'}
GRAPH_MODALITY_MODELS = {'MMGCN', 'GRCN', 'LATTICE', 'FREEDOM', 'LGMRec',
                         'BM3', 'SLMRec', 'MMSSL', 'DiffMM', 'MENTOR',
                         'DMRL', 'CLUSSL',
                         'MMSSL+MD', 'DiffMM+MD', 'LGMRec+MD', 'MENTOR+MD',
                         'MMSSL+DPR', 'DiffMM+DPR', 'LGMRec+DPR', 'MENTOR+DPR',
                         'MMSSL+FairDual', 'DiffMM+FairDual', 'LGMRec+FairDual', 'MENTOR+FairDual',
                         'MMSSL+Intra', 'DiffMM+Intra', 'LGMRec+Intra', 'MENTOR+Intra',
                         'MMSSL+Inter', 'DiffMM+Inter', 'LGMRec+Inter', 'MENTOR+Inter',
                         'MMSSL+Intra+Inter', 'DiffMM+Intra+Inter', 'LGMRec+Intra+Inter', 'MENTOR+Intra+Inter'}
I2MD4FAIR_MODELS = {'I2MD4Fair'}
MD_MODELS = {'MMSSL+MD', 'DiffMM+MD', 'LGMRec+MD', 'MENTOR+MD'}
DPR_MODELS = {'MMSSL+DPR', 'DiffMM+DPR', 'LGMRec+DPR', 'MENTOR+DPR'}
FAIRDUAL_MODELS = {'MMSSL+FairDual', 'DiffMM+FairDual', 'LGMRec+FairDual', 'MENTOR+FairDual'}
TRANSFER_MODELS = ({'MMSSL+Intra', 'DiffMM+Intra', 'LGMRec+Intra', 'MENTOR+Intra',
                    'MMSSL+Inter', 'DiffMM+Inter', 'LGMRec+Inter', 'MENTOR+Inter',
                    'MMSSL+Intra+Inter', 'DiffMM+Intra+Inter', 'LGMRec+Intra+Inter', 'MENTOR+Intra+Inter'})


def build_model(model_name, dataset, args, device):
    n_users = dataset.n_users
    n_items = dataset.n_items
    embed_dim = args.embed_dim
    modality_dims = dataset.get_modality_features_dim()

    base_model_name = model_name
    for suffix in ['+MD', '+DPR', '+FairDual', '+Intra+Inter', '+Intra', '+Inter']:
        if model_name.endswith(suffix):
            base_model_name = model_name[:-len(suffix)]
            break

    if base_model_name == 'I2MD4Fair':
        model = I2MD4Fair(
            n_users=n_users, n_items=n_items, embed_dim=embed_dim,
            modality_dims=modality_dims, n_protos=args.n_protos,
            eps=args.eps, p=args.p_norm, lam=args.lam, tau=args.tau,
            n_layers=args.n_layers, n_modality_layers=args.n_modality_layers,
            lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3
        )
    elif base_model_name == 'LightGCN':
        model = LightGCN(n_users, n_items, embed_dim, n_layers=args.n_layers)
    elif base_model_name == 'VBPR':
        model = VBPR(n_users, n_items, embed_dim, modality_dims['visual'])
    elif base_model_name == 'MMGCN':
        model = MMGCN(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif base_model_name == 'GRCN':
        model = GRCN(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif base_model_name == 'LATTICE':
        model = LATTICE(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif base_model_name == 'FREEDOM':
        model = FREEDOM(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif base_model_name == 'LGMRec':
        model = LGMRec(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif base_model_name == 'BM3':
        model = BM3(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif base_model_name == 'SLMRec':
        model = SLMRec(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif base_model_name == 'MMSSL':
        model = MMSSL(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif base_model_name == 'DiffMM':
        model = DiffMM(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif base_model_name == 'MENTOR':
        model = MENTOR(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif base_model_name == 'DMRL':
        model = DMRL(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    elif base_model_name == 'CLUSSL':
        model = CLUSSL(n_users, n_items, embed_dim, modality_dims, n_layers=args.n_layers)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name in MD_MODELS:
        model = ModalityDebiasingWrapper(model, n_users, n_items, embed_dim, modality_dims,
                                         model_type=base_model_name.lower())

    if model_name in DPR_MODELS:
        model = DPRWrapper(model, n_users, n_items, embed_dim, modality_dims)

    if model_name in FAIRDUAL_MODELS:
        model = FairDualWrapper(model, n_users, n_items, embed_dim, modality_dims)

    if model_name in TRANSFER_MODELS:
        use_intra = '+Intra' in model_name
        use_inter = '+Inter' in model_name
        model = BackboneWithDebias(
            model, modality_dims, embed_dim,
            use_intra=use_intra, use_inter=use_inter,
            n_protos=args.n_protos, eps=args.eps, p=args.p_norm,
            lam=args.lam, tau=args.tau
        )

    target = model.base_model if hasattr(model, 'base_model') else model
    if hasattr(target, 'set_precomputed_adj'):
        target.set_precomputed_adj(dataset.get_adj_matrices())

    return model.to(device)


def get_main_params(model, model_name):
    if model_name not in I2MD4FAIR_MODELS and model_name not in TRANSFER_MODELS:
        return list(model.parameters())
    club_params = set()
    if hasattr(model, 'inter_mdm') and hasattr(model.inter_mdm, 'club_estimators'):
        for k in model.inter_mdm.club_estimators:
            for p in model.inter_mdm.club_estimators[k].parameters():
                club_params.add(p)
    if hasattr(model, 'club_estimators'):
        for k in model.club_estimators:
            for p in model.club_estimators[k].parameters():
                club_params.add(p)
    main_params = [p for p in model.parameters() if p not in club_params]
    return main_params


def get_club_params(model, model_name):
    if model_name not in I2MD4FAIR_MODELS and model_name not in TRANSFER_MODELS:
        return []
    club_params = []
    if hasattr(model, 'inter_mdm') and hasattr(model.inter_mdm, 'club_estimators'):
        for k in model.inter_mdm.club_estimators:
            club_params.extend(list(model.inter_mdm.club_estimators[k].parameters()))
    if hasattr(model, 'club_estimators'):
        for k in model.club_estimators:
            club_params.extend(list(model.club_estimators[k].parameters()))
    return club_params


def train_one_epoch(model, dataset, data_loader, graph_norm, modality_features,
                    main_optimizer, club_optimizer, device, model_name,
                    inter_norm_u=None, inter_norm_v=None, epoch=0, warmup_epochs=5):
    model.train()
    total_loss = 0
    n_batches = 0
    is_warmup = epoch <= warmup_epochs

    for _ in range(len(data_loader)):
        user_ids, pos_ids, neg_ids = data_loader.get_batch()
        user_ids = user_ids.to(device)
        pos_ids = pos_ids.to(device)
        neg_ids = neg_ids.to(device)
        batch_item_ids = torch.unique(torch.cat([pos_ids, neg_ids]))

        needs_club = (model_name in I2MD4FAIR_MODELS or model_name in TRANSFER_MODELS) and club_optimizer is not None

        Z_I_full_dict = None
        if model_name in I2MD4FAIR_MODELS:
            Z_I_full_dict = model._compute_modality_reprs(
                modality_features, inter_norm_u, inter_norm_v, detach=True)

        if needs_club:
            club_optimizer.zero_grad()
            if model_name in I2MD4FAIR_MODELS:
                club_nll = model.club_nll_loss(
                    modality_features, graph_norm=graph_norm,
                    interaction_matrix_norm_u=inter_norm_u,
                    interaction_matrix_norm_v=inter_norm_v,
                    item_ids=batch_item_ids, Z_I_full_dict=Z_I_full_dict)
            else:
                club_nll = model.club_nll_loss(modality_features, batch_item_ids)
            club_nll.backward()
            torch.nn.utils.clip_grad_norm_(club_optimizer.param_groups[0]['params'], max_norm=5.0)
            club_optimizer.step()

        main_optimizer.zero_grad()

        if model_name in I2MD4FAIR_MODELS:
            loss, _, _, _ = model(graph_norm, modality_features, user_ids, pos_ids, neg_ids,
                                 interaction_matrix_norm_u=inter_norm_u,
                                 interaction_matrix_norm_v=inter_norm_v,
                                 warmup=is_warmup, Z_I_full_dict=Z_I_full_dict)
        elif model_name in TRANSFER_MODELS:
            loss = model.compute_loss(user_ids, pos_ids, neg_ids, graph_norm, modality_features,
                                      inter_norm_u=inter_norm_u, inter_norm_v=inter_norm_v,
                                      warmup=is_warmup)
        elif model_name in MODALITY_ONLY_MODELS:
            loss = model.compute_loss(user_ids, pos_ids, neg_ids, modality_features)
        elif model_name in GRAPH_ONLY_MODELS:
            loss = model.compute_loss(user_ids, pos_ids, neg_ids, graph_norm)
        elif model_name in GRAPH_MODALITY_MODELS:
            loss = model.compute_loss(user_ids, pos_ids, neg_ids, graph_norm, modality_features)
        else:
            raise ValueError(f"Unknown model category for: {model_name}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(main_optimizer.param_groups[0]['params'], max_norm=5.0)
        main_optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_and_eval(model_name, dataset, args, device, n_runs=5):
    results_all_runs = defaultdict(list)

    for run in range(n_runs):
        print(f"\n=== Run {run+1}/{n_runs} for {model_name} on {dataset.dataset_name} ===")
        torch.manual_seed(run)
        np.random.seed(run)
        model = build_model(model_name, dataset, args, device)

        main_params = get_main_params(model, model_name)
        main_optimizer = torch.optim.Adam(main_params, lr=args.lr)

        club_optimizer = None
        club_params_list = get_club_params(model, model_name)
        if club_params_list:
            club_optimizer = torch.optim.Adam(club_params_list, lr=args.lr)

        graph_norm = dataset.get_norm_graph().to(device)
        modality_features = dataset.get_modality_features()
        for k in modality_features:
            modality_features[k] = modality_features[k].to(device)

        inter_norm_u = None
        inter_norm_v = None
        if model_name in I2MD4FAIR_MODELS:
            norm_matrices = dataset.get_modality_norm_matrices()
            inter_norm_u = norm_matrices['inter_norm_u'].to(device)
            inter_norm_v = norm_matrices['inter_norm_v'].to(device)

        data_loader = BPRDataLoader(
            dataset.train_data, dataset.n_users, dataset.n_items,
            dataset.train_user_item_dict, batch_size=args.batch_size,
            user_item_dict=dataset.user_item_dict
        )

        best_recall10 = -1.0
        best_metrics = None
        best_state = None
        patience_counter = 0

        for epoch in range(1, args.max_epochs + 1):
            data_loader.shuffle()
            loss = train_one_epoch(model, dataset, data_loader, graph_norm,
                                   modality_features, main_optimizer, club_optimizer, device,
                                   model_name, inter_norm_u, inter_norm_v,
                                   epoch=epoch, warmup_epochs=args.warmup_epochs)

            if epoch % args.eval_interval == 0:
                metrics = evaluate_model(model, dataset, device=device, K_list=[10, 20], mode='val')
                recall10 = metrics['Recall'][10]
                ndcg10 = metrics['NDCG'][10]
                print(f"Epoch {epoch}: Loss={loss:.4f}, N@10={ndcg10:.4f}, "
                      f"R@10={recall10:.4f}, R@20={metrics['Recall'][20]:.4f}, "
                      f"HR@10={metrics.get('HR', {}).get(10, 0):.4f}, "
                      f"G@10={metrics['Gini'][10]:.4f}, E@10={metrics['Entropy'][10]:.4f}, "
                      f"C@10={metrics['Coverage'][10]:.4f}")

                if recall10 > best_recall10:
                    best_recall10 = recall10
                    best_metrics = metrics
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if best_metrics is not None:
            if best_state is not None:
                model.load_state_dict(best_state)
            test_metrics = evaluate_model(model, dataset, device=device, K_list=[10, 20], mode='test')
            for metric_name in test_metrics:
                for K in test_metrics[metric_name]:
                    results_all_runs[(metric_name, K)].append(test_metrics[metric_name][K])

            import json
            output_dir = f'results/{dataset.dataset_name}/{model_name}'
            os.makedirs(output_dir, exist_ok=True)
            seed_result = {'seed': run}
            for m in ['NDCG', 'Recall', 'HR', 'Gini', 'Entropy', 'Coverage', 'REG']:
                for K in [10, 20]:
                    if m in test_metrics and K in test_metrics[m]:
                        seed_result[f'{m}@{K}'] = test_metrics[m][K]
            with open(os.path.join(output_dir, f'seed_{run}.json'), 'w') as f:
                json.dump(seed_result, f, indent=2)

    avg_results = {}
    std_results = {}
    for key in results_all_runs:
        avg_results[key] = np.mean(results_all_runs[key])
        std_results[key] = np.std(results_all_runs[key], ddof=1) if len(results_all_runs[key]) > 1 else 0.0

    import json
    output_dir = f'results/{dataset.dataset_name}/{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        'mean': {f'{k[0]}@{k[1]}': v for k, v in avg_results.items()},
        'std': {f'{k[0]}@{k[1]}': v for k, v in std_results.items()},
        'n_runs': n_runs,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return avg_results


def run_single_experiment(args, device):
    dataset = DAMRSDataset(args.dataset, args.data_dir, args.embed_dim)

    results = train_and_eval(args.model, dataset, args, device, n_runs=args.n_runs)

    print(f"\n{'=' * 80}")
    print(f"Results for {args.model} on {args.dataset} (avg over {args.n_runs} runs)")
    print(f"{'=' * 80}")
    print(f"{'N@10':>8} {'N@20':>8} {'R@10':>8} {'R@20':>8} "
          f"{'HR@10':>8} {'HR@20':>8} "
          f"{'G@10':>8} {'G@20':>8} {'E@10':>8} {'E@20':>8} {'C@10':>8} {'C@20':>8} "
          f"{'REG@10':>8} {'REG@20':>8}")
    print(f"{results.get(('NDCG', 10), 0):>8.4f} {results.get(('NDCG', 20), 0):>8.4f} "
          f"{results.get(('Recall', 10), 0):>8.4f} {results.get(('Recall', 20), 0):>8.4f} "
          f"{results.get(('HR', 10), 0):>8.4f} {results.get(('HR', 20), 0):>8.4f} "
          f"{results.get(('Gini', 10), 0):>8.4f} {results.get(('Gini', 20), 0):>8.4f} "
          f"{results.get(('Entropy', 10), 0):>8.4f} {results.get(('Entropy', 20), 0):>8.4f} "
          f"{results.get(('Coverage', 10), 0):>8.4f} {results.get(('Coverage', 20), 0):>8.4f} "
          f"{results.get(('REG', 10), 0):>8.4f} {results.get(('REG', 20), 0):>8.4f}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='baby', choices=['baby', 'clothing', 'mind', 'demo'])
    parser.add_argument('--model', type=str, default='LightGCN',
                        choices=['I2MD4Fair', 'LightGCN', 'VBPR', 'MMGCN', 'GRCN',
                                 'LATTICE', 'FREEDOM', 'LGMRec', 'BM3', 'SLMRec',
                                 'MMSSL', 'DiffMM', 'MENTOR', 'DMRL', 'CLUSSL',
                                 'MMSSL+MD', 'DiffMM+MD', 'LGMRec+MD', 'MENTOR+MD',
                                 'MMSSL+DPR', 'DiffMM+DPR', 'LGMRec+DPR', 'MENTOR+DPR',
                                 'MMSSL+FairDual', 'DiffMM+FairDual', 'LGMRec+FairDual', 'MENTOR+FairDual',
                                 'MMSSL+Intra', 'DiffMM+Intra', 'LGMRec+Intra', 'MENTOR+Intra',
                                 'MMSSL+Inter', 'DiffMM+Inter', 'LGMRec+Inter', 'MENTOR+Inter',
                                 'MMSSL+Intra+Inter', 'DiffMM+Intra+Inter', 'LGMRec+Intra+Inter', 'MENTOR+Intra+Inter'])
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

    run_single_experiment(args, device)
