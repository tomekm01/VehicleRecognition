import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Subset

from models import create_standard_resnet, create_se_resnet, create_cbam_resnet
from datasets import CarsDataset, train_transform, val_transform

warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


DATASET_CONFIGS = {
    'stanford': {
        'csv_path': r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\StanfordCars_devkit\cars_train.csv",
        'img_dir':  r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\StanfordCars\cars_train\cars_train",
    },
    'compcars': {
        'csv_path':  r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\CompCars_devkit\compcars_dataset.csv",
        'img_dir':   r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\image",
        'train_txt': r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\train_test_split\classification\train.txt",
    },
}

MODEL_CONFIGS = {
    'standard': ('resnet', 'Standard ResNet50', create_standard_resnet),
    'se':       ('se',     'SE ResNet50',       create_se_resnet),
    'cbam':     ('cbam',   'CBAM ResNet50',     create_cbam_resnet),
}

_COLORS = {
    'Standard ResNet50': '#4C72B0',
    'SE ResNet50':       '#DD8452',
    'CBAM ResNet50':     '#55A868',
}
_MODEL_LABELS  = ['Standard ResNet50', 'SE ResNet50', 'CBAM ResNet50']
_MODEL_KEYS    = ['standard', 'se', 'cbam']
_BONFERRONI_A  = 0.05 / 3
_FIG_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
})


# ── Training & evaluation ────────────────────────────────────────────────────

def train_and_evaluate(model, train_loader, val_loader, device, epochs,
                       lr_base, lr_new, weight_decay, warmup_epochs):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    new_params, base_params = [], []
    for name, param in model.named_parameters():
        if 'fc' in name or 'se' in name or 'cbam' in name:
            new_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.Adam([
        {'params': base_params, 'lr': lr_base},
        {'params': new_params,  'lr': lr_new},
    ], weight_decay=weight_decay)

    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs)
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    top1_correct = top5_correct = total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.topk(5, 1, True, True)
            preds = preds.t()
            correct = preds.eq(labels.view(1, -1).expand_as(preds))
            top1_correct += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
            top5_correct += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            total += labels.size(0)
            all_preds.extend(preds[0].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return {
        'top1': top1_correct / total,
        'top5': top5_correct / total,
        'f1':   f1_score(all_labels, all_preds, average='macro'),
    }


def run_folds(create_fn, num_classes, train_base, val_base, folds, args,
              existing_results, on_fold_complete):
    results    = existing_results
    start_fold = len(results['top1'])
    total      = len(folds)

    if start_fold > 0:
        print(f"  Resuming from fold {start_fold + 1}/{total}")

    for fold_idx in range(start_fold, total):
        train_idx, val_idx = folds[fold_idx]
        print(f"  Fold {fold_idx + 1}/{total} ...", end='', flush=True)

        train_loader = DataLoader(
            Subset(train_base, train_idx),
            batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            Subset(val_base, val_idx),
            batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        metrics = train_and_evaluate(
            create_fn(num_classes), train_loader, val_loader, args.device, args.epochs,
            lr_base=args.lr_base, lr_new=args.lr_new,
            weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs
        )
        for k in results:
            results[k].append(metrics[k])
        on_fold_complete(results)
        print(f"  Top-1: {metrics['top1']:.4f} | Top-5: {metrics['top5']:.4f} | F1: {metrics['f1']:.4f}")

    return results


# ── Statistical analysis ─────────────────────────────────────────────────────

def _safe_wilcoxon(x, y):
    if np.allclose(np.array(x) - np.array(y), 0):
        return 0.0, 1.0
    return wilcoxon(x, y)


def run_statistics(res_std, res_se, res_cbam, metric_name):
    lines = [f"\n--- Statistical Analysis ({metric_name}) ---"]
    stat, p = friedmanchisquare(res_std, res_se, res_cbam)
    lines.append(f"Friedman Test: Statistic={stat:.4f}, p-value={p:.4e}")

    if p < 0.05:
        alpha = _BONFERRONI_A
        lines.append(f"Result: Significant differences found. Running Wilcoxon post-hoc...")
        lines.append(f"Bonferroni-corrected alpha: {alpha:.4f}")
        _, p1 = _safe_wilcoxon(res_std, res_se)
        _, p2 = _safe_wilcoxon(res_std, res_cbam)
        _, p3 = _safe_wilcoxon(res_se,  res_cbam)
        lines.append(f"  Std vs SE:   p={p1:.4e}  {'(Significant)' if p1 < alpha else '(Not Sig)'}")
        lines.append(f"  Std vs CBAM: p={p2:.4e}  {'(Significant)' if p2 < alpha else '(Not Sig)'}")
        lines.append(f"  SE vs CBAM:  p={p3:.4e}  {'(Significant)' if p3 < alpha else '(Not Sig)'}")
    else:
        lines.append("Result: No statistically significant differences found (p >= 0.05).")

    return "\n".join(lines)


def try_run_full_statistics(args):
    result_files = {m: f'results_{m}.json' for m in _MODEL_KEYS}
    missing = [m for m, f in result_files.items() if not os.path.exists(f)]
    if missing:
        print(f"\nStatistical analysis pending — still waiting for: {missing}")
        return

    print("\nAll 3 model results found — running full statistical analysis...")
    data = {}
    for m, fp in result_files.items():
        with open(fp) as f:
            data[m] = json.load(f)

    lines = ["=" * 60, "PHASE 1 — FULL STATISTICAL ANALYSIS", "=" * 60]
    for dataset in ['stanford', 'compcars']:
        lines += [f"\n{'='*60}", f"Dataset: {dataset.upper()}", f"{'='*60}"]
        lines.append(f"\n{'Model':<22} {'Top-1':>8} {'Top-5':>8} {'Macro-F1':>10}")
        lines.append("-" * 52)
        for m in _MODEL_KEYS:
            r     = data[m]['results'][dataset]
            label = data[m]['model_label']
            lines.append(f"{label:<22} {np.mean(r['top1']):>8.4f} "
                         f"{np.mean(r['top5']):>8.4f} {np.mean(r['f1']):>10.4f}")

        lines.append("\nPer-fold results:")
        n_folds = len(data['standard']['results'][dataset]['top1'])
        for i in range(n_folds):
            lines.append(f"  Fold {i+1}:")
            for m in _MODEL_KEYS:
                r     = data[m]['results'][dataset]
                label = data[m]['model_label']
                lines.append(f"    {label:<20} Top-1={r['top1'][i]:.4f}  "
                             f"Top-5={r['top5'][i]:.4f}  F1={r['f1'][i]:.4f}")

        for metric in ['top1', 'f1']:
            label = 'Top-1 Accuracy' if metric == 'top1' else 'Macro F1-Score'
            lines.append(run_statistics(
                data['standard']['results'][dataset][metric],
                data['se']['results'][dataset][metric],
                data['cbam']['results'][dataset][metric],
                label
            ))

    output = "\n".join(lines)
    print(output)
    with open('phase1_analysis.txt', 'w') as f:
        f.write(output)
    print("\nFull analysis saved to phase1_analysis.txt")


# ── Figures ──────────────────────────────────────────────────────────────────

def _save_fig(fig, name):
    os.makedirs(_FIG_DIR, exist_ok=True)
    base = os.path.join(_FIG_DIR, name)
    fig.savefig(base + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(base + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved figures/{name}.pdf  +  .png')


def _ds_title(dataset):
    return 'Stanford Cars' if dataset == 'stanford' else 'CompCars'


def _plot_boxplot(data, dataset):
    np.random.seed(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric, title in zip(axes, ['top1', 'f1'], ['Top-1 Accuracy', 'Macro F1']):
        vals = [data[mk]['results'][dataset][metric] for mk in _MODEL_KEYS]
        bp = ax.boxplot(
            vals, patch_artist=True, widths=0.48,
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
            flierprops=dict(marker=''),
        )
        for patch, label in zip(bp['boxes'], _MODEL_LABELS):
            patch.set_facecolor(_COLORS[label]); patch.set_alpha(0.55); patch.set_linewidth(1.2)
        for i, (v, label) in enumerate(zip(vals, _MODEL_LABELS)):
            jitter = np.random.normal(i + 1, 0.06, len(v))
            ax.scatter(jitter, v, s=22, alpha=0.65, color=_COLORS[label],
                       edgecolors='white', linewidths=0.4, zorder=3)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(_MODEL_LABELS, rotation=13, ha='right')
        ax.set_ylabel('Score'); ax.set_title(title)
        ax.yaxis.grid(True, linestyle=':', alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle(f'Phase 1 — {_ds_title(dataset)}: Per-Fold Score Distribution',
                 fontweight='bold')
    fig.tight_layout()
    _save_fig(fig, f'phase1_boxplot_{dataset}')


def _plot_bars(data, dataset):
    metrics = ['top1', 'top5', 'f1']
    x, width = np.arange(3), 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (mk, label) in enumerate(zip(_MODEL_KEYS, _MODEL_LABELS)):
        means = [np.mean(data[mk]['results'][dataset][m]) for m in metrics]
        stds  = [np.std( data[mk]['results'][dataset][m]) for m in metrics]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=label,
               color=_COLORS[label], capsize=4, alpha=0.85, linewidth=0.8,
               error_kw=dict(linewidth=1.2, capthick=1.2))
    all_vals = [v for mk in _MODEL_KEYS for m in metrics
                for v in data[mk]['results'][dataset][m]]
    ax.set_ylim(bottom=max(0.0, min(all_vals) - 0.08))
    ax.set_xticks(x); ax.set_xticklabels(['Top-1', 'Top-5', 'Macro F1'])
    ax.set_ylabel('Score')
    ax.yaxis.grid(True, linestyle=':', alpha=0.5); ax.set_axisbelow(True)
    ax.legend(framealpha=0.9)
    ax.set_title(f'Phase 1 — {_ds_title(dataset)}: Mean Scores (±1 std, 10 folds)')
    fig.tight_layout()
    _save_fig(fig, f'phase1_bars_{dataset}')


def _plot_wilcoxon(data, dataset):
    pairs = [
        ('standard', 'se',   'Std vs SE'),
        ('standard', 'cbam', 'Std vs CBAM'),
        ('se',       'cbam', 'SE vs CBAM'),
    ]
    pvals = np.zeros((3, 2))
    for i, (m1, m2, _) in enumerate(pairs):
        for j, metric in enumerate(['top1', 'f1']):
            x = data[m1]['results'][dataset][metric]
            y = data[m2]['results'][dataset][metric]
            _, pvals[i, j] = _safe_wilcoxon(x, y)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(pvals, cmap='RdYlGn', vmin=0.0, vmax=0.10, aspect='auto')
    for i in range(3):
        for j in range(2):
            p   = pvals[i, j]
            sig = '*' if p < _BONFERRONI_A else 'ns'
            ax.text(j, i, f'p={p:.4f}\n{sig}',
                    ha='center', va='center', fontsize=8.5,
                    color='white' if p < 0.025 else 'black')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Top-1', 'Macro F1'])
    ax.set_yticks(range(3)); ax.set_yticklabels([p[2] for p in pairs])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('p-value', fontsize=9)
    ax.set_title(
        f'Phase 1 — {_ds_title(dataset)}: Wilcoxon p-values\n'
        f'(* = Bonferroni-corrected alpha={_BONFERRONI_A:.4f})'
    )
    fig.tight_layout()
    _save_fig(fig, f'phase1_wilcoxon_{dataset}')


def generate_phase1_plots(data):
    for dataset in ['stanford', 'compcars']:
        print(f'\nGenerating Phase 1 figures for {dataset}...')
        _plot_boxplot(data, dataset)
        _plot_bars(data, dataset)
        _plot_wilcoxon(data, dataset)


# ── CLI & main ───────────────────────────────────────────────────────────────

def _phase1_complete(total_folds):
    for m in _MODEL_KEYS:
        f = f'results_{m}.json'
        if not os.path.exists(f):
            return False
        with open(f) as fh:
            d = json.load(fh)
        for ds in ['stanford', 'compcars']:
            r = d.get('results', {}).get(ds, {})
            if len(r.get('top1', [])) < total_folds:
                return False
    return True


def get_config():
    parser = argparse.ArgumentParser(
        description="Phase 1: per-model RSKF experiment across both datasets"
    )
    parser.add_argument('--model',         type=str, required=True,
                        choices=['standard', 'se', 'cbam'])
    parser.add_argument('--batch_size',    type=int,   default=16)
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--folds',         type=int,   default=5)
    parser.add_argument('--repeats',       type=int,   default=2)
    parser.add_argument('--lr_new',        type=float, default=1e-3)
    parser.add_argument('--lr_base',       type=float, default=1e-5)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int,   default=5)
    parser.add_argument('--seed',          type=int,   default=21,
                        help="Must be identical across all 3 model runs")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = get_config()
    set_seed(args.seed)

    total_folds = args.folds * args.repeats

    if _phase1_complete(total_folds):
        print("All Phase 1 results complete — generating figures (no training).")
        data = {}
        for m in _MODEL_KEYS:
            with open(f'results_{m}.json') as fh:
                data[m] = json.load(fh)
        generate_phase1_plots(data)
        print(f"\nDone. Figures saved to {_FIG_DIR}")
        return

    _, model_label, create_fn = MODEL_CONFIGS[args.model]

    print(f"Device:      {args.device}")
    print(f"Model:       {model_label}")
    print(f"Epochs:      {args.epochs}  |  Batch: {args.batch_size}  |  Seed: {args.seed}")
    print(f"Folds:       {args.folds}x{args.repeats} = {total_folds} iterations per dataset")
    print(f"LR base:     {args.lr_base}  |  LR new: {args.lr_new}  |  WD: {args.weight_decay}"
          f"  |  Warmup: {args.warmup_epochs}\n")

    result_file = f'results_{args.model}.json'

    if os.path.exists(result_file):
        with open(result_file) as f:
            saved = json.load(f)
        all_results = saved.get('results', {})
        for ds, r in all_results.items():
            n = len(r['top1'])
            status = 'complete' if n >= total_folds else f'{n}/{total_folds} folds done'
            print(f"  Loaded {ds}: {status}")
        print()
    else:
        all_results = {}

    config_block = {
        'epochs': args.epochs, 'folds': args.folds, 'repeats': args.repeats,
        'seed': args.seed, 'lr_new': args.lr_new, 'lr_base': args.lr_base,
        'weight_decay': args.weight_decay, 'warmup_epochs': args.warmup_epochs,
        'batch_size': args.batch_size,
    }

    def save_fold(dataset_name, partial_results):
        all_results[dataset_name] = partial_results
        with open(result_file, 'w') as f:
            json.dump({
                'model': args.model, 'model_label': model_label,
                'config': config_block, 'results': all_results,
            }, f, indent=2)

    for dataset_name, cfg in DATASET_CONFIGS.items():
        existing   = all_results.get(dataset_name, {'top1': [], 'top5': [], 'f1': []})
        done_folds = len(existing['top1'])

        if done_folds >= total_folds:
            print(f"\nSkipping {dataset_name.upper()} — all {total_folds} folds complete.")
            continue

        print(f"\n{'='*55}\nDataset: {dataset_name.upper()}\n{'='*55}")

        df = pd.read_csv(cfg['csv_path'])
        if dataset_name == 'compcars':
            with open(cfg['train_txt']) as f:
                valid = {l.strip() for l in f}
            df = df[df['filename'].isin(valid)].reset_index(drop=True)
            print(f"Filtered to {len(df)} official training images.")

        le = LabelEncoder()
        df['mapped_label'] = le.fit_transform(df['class_id'])
        num_classes = len(le.classes_)
        print(f"Classes: {num_classes}")

        train_base = CarsDataset(df, cfg['img_dir'], transform=train_transform)
        val_base   = CarsDataset(df, cfg['img_dir'], transform=val_transform)
        targets    = df['mapped_label'].values

        rskf  = RepeatedStratifiedKFold(
            n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed
        )
        folds = list(rskf.split(np.zeros(len(targets)), targets))

        results = run_folds(
            create_fn, num_classes, train_base, val_base, folds, args,
            existing_results=existing,
            on_fold_complete=lambda r, d=dataset_name: save_fold(d, r),
        )
        all_results[dataset_name] = results
        print(f"\n  {dataset_name.upper()} averages:")
        print(f"    Top-1: {np.mean(results['top1']):.4f}  "
              f"Top-5: {np.mean(results['top5']):.4f}  "
              f"F1: {np.mean(results['f1']):.4f}")

    print(f"\nResults saved to {result_file}")
    try_run_full_statistics(args)


if __name__ == '__main__':
    main()
