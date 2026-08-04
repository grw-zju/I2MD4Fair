import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_seeds(results_dir, model, dataset):
    model_dir = os.path.join(results_dir, dataset, model)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"No results directory: {model_dir}")
    seeds = {}
    for f in sorted(os.listdir(model_dir)):
        if f.startswith('seed_') and f.endswith('.json'):
            with open(os.path.join(model_dir, f)) as fh:
                data = json.load(fh)
            seeds[data['seed']] = data
    return seeds


def run_stats(args):
    our = load_seeds(args.results_dir, args.our_model, args.dataset)
    comp = load_seeds(args.results_dir, args.comparator, args.dataset)

    our_seeds = set(our.keys())
    comp_seeds = set(comp.keys())
    if our_seeds != comp_seeds:
        missing = our_seeds.symmetric_difference(comp_seeds)
        raise ValueError(f"Unmatched seeds: {missing}")

    metrics = ['Recall@10', 'NDCG@10', 'HR@10', 'Gini@10', 'Entropy@10', 'Coverage@10', 'REG@10']
    sorted_seeds = sorted(our_seeds)

    from scipy import stats

    print(f"{'Metric':<15} {'t_stat':>10} {'p_value':>12} {'diff_mean':>12} "
          f"{'diff_std':>10} {'CI_low':>10} {'CI_high':>10} {'effect_size':>12}")
    print("=" * 95)

    p_values = {}
    for metric in metrics:
        our_vals = np.array([our[s][metric] for s in sorted_seeds])
        comp_vals = np.array([comp[s][metric] for s in sorted_seeds])
        diff = our_vals - comp_vals
        n = len(diff)

        t_stat, p_value = stats.ttest_rel(our_vals, comp_vals)
        mean_diff = diff.mean()
        se_diff = diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        ci_low = mean_diff - stats.t.ppf(0.975, n - 1) * se_diff if n > 1 else mean_diff
        ci_high = mean_diff + stats.t.ppf(0.975, n - 1) * se_diff if n > 1 else mean_diff
        effect_size = mean_diff / (diff.std(ddof=1) + 1e-10) if n > 1 else 0.0

        p_values[metric] = p_value
        print(f"{metric:<15} {t_stat:>10.4f} {p_value:>12.6f} {mean_diff:>12.4f} "
              f"{diff.std(ddof=1):>10.4f} {ci_low:>10.4f} {ci_high:>10.4f} {effect_size:>12.4f}")

    sorted_metrics = sorted(metrics, key=lambda m: p_values[m])
    n_tests = len(sorted_metrics)
    print(f"\nHolm-adjusted p-values:")
    prev_adjusted = 0.0
    for rank, metric in enumerate(sorted_metrics, 1):
        adjusted = min(p_values[metric] * (n_tests - rank + 1), 1.0)
        adjusted = max(adjusted, prev_adjusted)
        prev_adjusted = adjusted
        print(f"  {metric}: raw={p_values[metric]:.6f}, adjusted={adjusted:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paired t-test between two models')
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--our_model', type=str, required=True)
    parser.add_argument('--comparator', type=str, required=True)
    args = parser.parse_args()
    run_stats(args)
