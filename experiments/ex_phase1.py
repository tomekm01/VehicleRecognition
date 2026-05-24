import argparse
import json
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from scipy.stats import friedmanchisquare, wilcoxon
import numpy as np
import random
import warnings
from sklearn.preprocessing import LabelEncoder

from models import create_standard_resnet, create_se_resnet, create_cbam_resnet
from datasets import CarsDataset, train_transform, val_transform

warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


# Dataset configs - both datasets always run for the selected model

DATASET_CONFIGS = {
    'stanford': {
        'csv_path':  r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\StanfordCars_devkit\cars_train.csv",
        'img_dir':   r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\StanfordCars\cars_train\cars_train",
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


# Training & evaluation

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
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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

    metrics = {
        'top1': top1_correct / total,
        'top5': top5_correct / total,
        'f1':   f1_score(all_labels, all_preds, average='macro'),
    }

    del model
    torch.cuda.empty_cache()
    return metrics


# Statistical analysis

def run_statistics(res_std, res_se, res_cbam, metric_name):
    output = [f"\n--- Statistical Analysis ({metric_name}) ---"]

    stat, p_friedman = friedmanchisquare(res_std, res_se, res_cbam)
    output.append(f"Friedman Test: Statistic={stat:.4f}, p-value={p_friedman:.4e}")

    if p_friedman < 0.05:
        output.append("Result: Significant differences found. Running Wilcoxon post-hoc...")
        alpha_corrected = 0.05 / 3
        output.append(f"Bonferroni-corrected alpha: {alpha_corrected:.4f}")

        def safe_wilcoxon(x, y):
            if np.allclose(np.array(x) - np.array(y), 0):
                return 0.0, 1.0
            return wilcoxon(x, y)

        _, p1 = safe_wilcoxon(res_std, res_se)
        _, p2 = safe_wilcoxon(res_std, res_cbam)
        _, p3 = safe_wilcoxon(res_se,  res_cbam)

        output.append(f"  Std vs SE:   p={p1:.4e}  {'(Significant)' if p1 < alpha_corrected else '(Not Sig)'}")
        output.append(f"  Std vs CBAM: p={p2:.4e}  {'(Significant)' if p2 < alpha_corrected else '(Not Sig)'}")
        output.append(f"  SE vs CBAM:  p={p3:.4e}  {'(Significant)' if p3 < alpha_corrected else '(Not Sig)'}")
    else:
        output.append("Result: No statistically significant differences found (p >= 0.05).")

    return "\n".join(output)


# Fold loop with per-fold saving.

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


# Full statistical analysis (runs when all 3 model result files exist)

def try_run_full_statistics(args):
    result_files = {m: f'results_{m}.json' for m in ['standard', 'se', 'cbam']}
    missing = [m for m, f in result_files.items() if not os.path.exists(f)]

    if missing:
        print(f"\nStatistical analysis pending — still waiting for: {missing}")
        return

    print("\nAll 3 model results found — running full statistical analysis...")

    data = {}
    for model_name, filepath in result_files.items():
        with open(filepath, 'r') as f:
            data[model_name] = json.load(f)

    lines = ["=" * 60, "PHASE 1 — FULL STATISTICAL ANALYSIS", "=" * 60]

    for dataset in ['stanford', 'compcars']:
        lines.append(f"\n{'='*60}")
        lines.append(f"Dataset: {dataset.upper()}")
        lines.append(f"{'='*60}")

        # Average scores table
        lines.append(f"\n{'Model':<22} {'Top-1':>8} {'Top-5':>8} {'Macro-F1':>10}")
        lines.append("-" * 52)
        for model_name in ['standard', 'se', 'cbam']:
            r = data[model_name]['results'][dataset]
            label = data[model_name]['model_label']
            lines.append(
                f"{label:<22} {np.mean(r['top1']):>8.4f} "
                f"{np.mean(r['top5']):>8.4f} {np.mean(r['f1']):>10.4f}"
            )

        # Per-fold results
        lines.append("\nPer-fold results:")
        n_folds = len(data['standard']['results'][dataset]['top1'])
        for i in range(n_folds):
            lines.append(f"  Fold {i+1}:")
            for model_name in ['standard', 'se', 'cbam']:
                r = data[model_name]['results'][dataset]
                label = data[model_name]['model_label']
                lines.append(
                    f"    {label:<20} Top-1={r['top1'][i]:.4f}  "
                    f"Top-5={r['top5'][i]:.4f}  F1={r['f1'][i]:.4f}"
                )

        # Friedman + Wilcoxon
        for metric in ['top1', 'f1']:
            metric_label = 'Top-1 Accuracy' if metric == 'top1' else 'Macro F1-Score'
            lines.append(run_statistics(
                data['standard']['results'][dataset][metric],
                data['se']['results'][dataset][metric],
                data['cbam']['results'][dataset][metric],
                metric_label
            ))

    output = "\n".join(lines)
    print(output)

    analysis_file = 'phase1_analysis.txt'
    with open(analysis_file, 'w') as f:
        f.write(output)
    print(f"\nFull analysis saved to {analysis_file}")


# Config

def get_config():
    parser = argparse.ArgumentParser(
        description="Phase 1: per-model RSKF experiment across both datasets"
    )
    parser.add_argument('--model',        type=str, required=True,
                        choices=['standard', 'se', 'cbam'],
                        help="Which model to run. Each model run covers both datasets.")
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--folds',        type=int,   default=5)
    parser.add_argument('--repeats',      type=int,   default=2)
    parser.add_argument('--lr_new',       type=float, default=1e-3)
    parser.add_argument('--lr_base',      type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs',type=int,   default=5)
    parser.add_argument('--seed',         type=int,   default=21,
                        help="Random seed — must be identical across all 3 model runs")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Main

def main():
    args = get_config()
    set_seed(args.seed)

    _, model_label, create_fn = MODEL_CONFIGS[args.model]

    print(f"Device:      {args.device}")
    print(f"Model:       {model_label}")
    print(f"Epochs:      {args.epochs}  |  Batch: {args.batch_size}  |  Seed: {args.seed}")
    print(f"Folds:       {args.folds}x{args.repeats} = {args.folds * args.repeats} iterations per dataset")
    print(f"LR base:     {args.lr_base}  |  LR new: {args.lr_new}  |  WD: {args.weight_decay}  |  Warmup: {args.warmup_epochs}\n")

    result_file = f'results_{args.model}.json'
    total_folds = args.folds * args.repeats

    # Load any fold results already written in a previous session.

    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
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
        'epochs':        args.epochs,
        'folds':         args.folds,
        'repeats':       args.repeats,
        'seed':          args.seed,
        'lr_new':        args.lr_new,
        'lr_base':       args.lr_base,
        'weight_decay':  args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'batch_size':    args.batch_size,
    }

    def save_fold(dataset_name, partial_results):
        all_results[dataset_name] = partial_results
        with open(result_file, 'w') as f:
            json.dump({
                'model':       args.model,
                'model_label': model_label,
                'config':      config_block,
                'results':     all_results,
            }, f, indent=2)

    for dataset_name, cfg in DATASET_CONFIGS.items():
        existing   = all_results.get(dataset_name, {'top1': [], 'top5': [], 'f1': []})
        done_folds = len(existing['top1'])

        if done_folds >= total_folds:
            print(f"\nSkipping {dataset_name.upper()} — all {total_folds} folds complete.")
            continue

        print(f"\n{'='*55}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*55}")

        df = pd.read_csv(cfg['csv_path'])

        if dataset_name == 'compcars':
            with open(cfg['train_txt'], 'r') as f:
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

    # If all 3 model files exist, run the full statistical analysis automatically
    try_run_full_statistics(args)


if __name__ == '__main__':
    main()
