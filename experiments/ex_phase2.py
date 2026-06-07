import argparse
import json
import os
from pathlib import Path

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
from PIL import Image
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, Subset

from ultralytics import YOLO

from models import create_standard_resnet, create_se_resnet, create_cbam_resnet
from datasets import CarsDataset, train_transform, val_transform

warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


_COLORS = {
    'Standard ResNet50': '#4C72B0',
    'SE ResNet50':       '#DD8452',
    'CBAM ResNet50':     '#55A868',
}
_MODEL_LABELS = ['Standard ResNet50', 'SE ResNet50', 'CBAM ResNet50']
_MODEL_KEYS   = ['resnet', 'se', 'cbam']
_BONFERRONI_A = 0.05 / 3
_FIG_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
})


# ── YOLO crops dataset ───────────────────────────────────────────────────────

class YOLODataset(Dataset):
    def __init__(self, pil_images, labels, transform=None):
        self.images    = pil_images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── YOLO box caching ─────────────────────────────────────────────────────────

def load_or_build_yolo_dataset(test_df, test_dir, yolo_model_name, cache_path):
    vehicle_classes = {2, 5, 7}
    n = len(test_df)

    def apply_boxes(boxes):
        crops = []
        for (_, row), box in zip(test_df.iterrows(), boxes):
            img = Image.open(Path(test_dir) / row['filename']).convert('RGB')
            if box is not None:
                x1, y1, x2, y2 = box
                w, h = img.size
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    img = img.crop((x1, y1, x2, y2))
            crops.append(img)
        return crops

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        if cache.get('n_images') == n:
            print(f"  YOLO cache loaded from {cache_path}  "
                  f"(fallbacks: {cache['fallback_count']}/{n})")
            return apply_boxes(cache['boxes']), test_df['mapped_label'].tolist()

    print(f"  Running YOLO ({yolo_model_name}) on {n} test images...")
    yolo = YOLO(yolo_model_name)
    boxes, fallbacks = [], 0

    for i, (_, row) in enumerate(test_df.iterrows()):
        img_path   = str(Path(test_dir) / row['filename'])
        detections = yolo(img_path, verbose=False)[0]
        vboxes     = [b for b in detections.boxes if int(b.cls) in vehicle_classes]
        if vboxes:
            best = max(vboxes, key=lambda b: float(b.conf))
            boxes.append(list(map(int, best.xyxy[0].cpu().numpy())))
        else:
            boxes.append(None)
            fallbacks += 1
        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"    {i+1}/{n}  (fallbacks so far: {fallbacks})")

    del yolo
    torch.cuda.empty_cache()

    with open(cache_path, 'w') as f:
        json.dump({'n_images': n, 'fallback_count': fallbacks, 'boxes': boxes}, f)
    print(f"  YOLO complete. Fallbacks: {fallbacks}/{n}. Cache saved to {cache_path}")

    return apply_boxes(boxes), test_df['mapped_label'].tolist()


# ── Training & evaluation ────────────────────────────────────────────────────

def train_model(model, train_loader, device, epochs,
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

    model.cpu()
    torch.cuda.empty_cache()
    return model


def evaluate_model(model, loader, device):
    model = model.to(device)
    model.eval()
    top1_correct = top5_correct = total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds = model(inputs).topk(5, 1, True, True)
            preds  = preds.t()
            correct = preds.eq(labels.view(1, -1).expand_as(preds))
            top1_correct += correct[:1].reshape(-1).float().sum().item()
            top5_correct += correct[:5].reshape(-1).float().sum().item()
            total += labels.size(0)
            all_preds.extend(preds[0].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    model.cpu()
    torch.cuda.empty_cache()
    return {
        'top1': top1_correct / total,
        'top5': top5_correct / total,
        'f1':   f1_score(all_labels, all_preds, average='macro'),
    }


def run_model_folds(create_fn, num_classes, train_base, folds,
                    gt_loader, yolo_loader, args,
                    existing_results, on_fold_complete):
    results    = existing_results
    start_fold = len(results['gt']['top1'])
    total      = len(folds)

    if start_fold > 0:
        print(f"  Resuming from fold {start_fold + 1}/{total}")

    for fold_idx in range(start_fold, total):
        train_idx, _ = folds[fold_idx]
        print(f"  Fold {fold_idx + 1}/{total} — training...", end='', flush=True)

        train_loader = DataLoader(
            Subset(train_base, train_idx),
            batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        model = train_model(
            create_fn(num_classes), train_loader, args.device, args.epochs,
            args.lr_base, args.lr_new, args.weight_decay, args.warmup_epochs
        )
        m_gt   = evaluate_model(model, gt_loader,   args.device)
        m_yolo = evaluate_model(model, yolo_loader, args.device)
        del model
        torch.cuda.empty_cache()

        for k in ('top1', 'top5', 'f1'):
            results['gt'][k].append(m_gt[k])
            results['yolo'][k].append(m_yolo[k])
        on_fold_complete(results)

        print(f"  done")
        print(f"    GT:   Top-1={m_gt['top1']:.4f}  Top-5={m_gt['top5']:.4f}  F1={m_gt['f1']:.4f}")
        print(f"    YOLO: Top-1={m_yolo['top1']:.4f}  Top-5={m_yolo['top5']:.4f}  F1={m_yolo['f1']:.4f}")
        print(f"    Top-1 drop (GT-YOLO): {m_gt['top1'] - m_yolo['top1']:+.4f}")

    return results


# ── Statistical analysis ─────────────────────────────────────────────────────

def _safe_wilcoxon(x, y):
    if np.allclose(np.array(x) - np.array(y), 0):
        return 0.0, 1.0
    return wilcoxon(x, y)


def run_stats_3models(res_std, res_se, res_cbam, metric, condition):
    out = [f"\n--- {condition} | {metric} ---"]
    stat, p = friedmanchisquare(res_std, res_se, res_cbam)
    out.append(f"Friedman: statistic={stat:.4f}, p={p:.4e}")
    if p < 0.05:
        alpha = _BONFERRONI_A
        out.append(f"Significant — Wilcoxon post-hoc (Bonferroni alpha={alpha:.4f}):")
        _, p1 = _safe_wilcoxon(res_std, res_se)
        _, p2 = _safe_wilcoxon(res_std, res_cbam)
        _, p3 = _safe_wilcoxon(res_se,  res_cbam)
        out.append(f"  Std vs SE:   p={p1:.4e}  {'(Sig)' if p1 < alpha else '(Not Sig)'}")
        out.append(f"  Std vs CBAM: p={p2:.4e}  {'(Sig)' if p2 < alpha else '(Not Sig)'}")
        out.append(f"  SE vs CBAM:  p={p3:.4e}  {'(Sig)' if p3 < alpha else '(Not Sig)'}")
    else:
        out.append("No significant differences (p >= 0.05).")
    return "\n".join(out)


def run_stats_gt_vs_yolo(gt, yolo, model_name, metric):
    gt_arr, yolo_arr = np.array(gt), np.array(yolo)
    delta = gt_arr - yolo_arr
    out = [f"\n--- GT vs YOLO | {model_name} | {metric} ---"]
    out.append(f"  GT   mean: {gt_arr.mean():.4f}  (+/-{gt_arr.std():.4f})")
    out.append(f"  YOLO mean: {yolo_arr.mean():.4f}  (+/-{yolo_arr.std():.4f})")
    out.append(f"  Mean accuracy drop (GT - YOLO): {delta.mean():+.4f}")
    if np.allclose(delta, 0):
        out.append("  Wilcoxon: all deltas zero, p=1.0 (Not Significant)")
    else:
        _, p = wilcoxon(gt_arr, yolo_arr)
        out.append(f"  Wilcoxon p={p:.4e}  {'(Significant)' if p < 0.05 else '(Not Significant)'}")
    return "\n".join(out)


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


def _plot_bars(state, dataset):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric, title in zip(axes, ['top1', 'f1'], ['Top-1 Accuracy', 'Macro F1']):
        x, width = np.arange(3), 0.38
        colors = [_COLORS[l] for l in _MODEL_LABELS]
        gt_means   = [np.mean(state[mk]['gt'][metric])   for mk in _MODEL_KEYS]
        gt_stds    = [np.std( state[mk]['gt'][metric])   for mk in _MODEL_KEYS]
        yolo_means = [np.mean(state[mk]['yolo'][metric]) for mk in _MODEL_KEYS]
        yolo_stds  = [np.std( state[mk]['yolo'][metric]) for mk in _MODEL_KEYS]
        ax.bar(x - width / 2, gt_means, width, yerr=gt_stds,
               color=colors, alpha=0.88, capsize=4, linewidth=0.8,
               error_kw=dict(linewidth=1.2, capthick=1.2))
        ax.bar(x + width / 2, yolo_means, width, yerr=yolo_stds,
               color=colors, alpha=0.38, capsize=4, hatch='//', linewidth=0.8,
               error_kw=dict(linewidth=1.2, capthick=1.2))
        ax.set_xticks(x)
        ax.set_xticklabels(_MODEL_LABELS, rotation=13, ha='right')
        ax.set_ylabel('Score'); ax.set_title(title)
        ax.set_ylim(bottom=max(0.0, min(gt_means + yolo_means) - 0.08))
        ax.yaxis.grid(True, linestyle=':', alpha=0.5); ax.set_axisbelow(True)
        gt_p   = mpatches.Patch(facecolor='#888', alpha=0.88, label='GT (bbox)')
        yolo_p = mpatches.Patch(facecolor='#888', alpha=0.38, hatch='//', label='YOLO crop')
        ax.legend(handles=[gt_p, yolo_p], framealpha=0.9)
    fig.suptitle(f'Phase 2 — {_ds_title(dataset)}: GT vs YOLO Detection', fontweight='bold')
    fig.tight_layout()
    _save_fig(fig, f'phase2_bars_{dataset}')


def _plot_dumbbell(state, dataset):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, metric, title in zip(axes, ['top1', 'f1'], ['Top-1 Accuracy', 'Macro F1']):
        for i, (mk, label) in enumerate(zip(_MODEL_KEYS, _MODEL_LABELS)):
            gt_mean   = np.mean(state[mk]['gt'][metric])
            yolo_mean = np.mean(state[mk]['yolo'][metric])
            color = _COLORS[label]
            ax.plot([gt_mean, yolo_mean], [i, i],
                    color=color, lw=2.5, zorder=2, solid_capstyle='round')
            ax.scatter([gt_mean],   [i], color=color, s=120, zorder=4)
            ax.scatter([yolo_mean], [i], facecolors='white', edgecolors=color,
                       s=120, zorder=4, linewidths=2.2)
            ax.annotate(
                f'{gt_mean - yolo_mean:+.3f}',
                xy=((gt_mean + yolo_mean) / 2, i), xytext=(0, 9),
                textcoords='offset points',
                ha='center', va='bottom', fontsize=8, color=color,
            )
        ax.set_yticks(range(3)); ax.set_yticklabels(_MODEL_LABELS)
        ax.set_xlabel('Score'); ax.set_title(title)
        ax.xaxis.grid(True, linestyle=':', alpha=0.5); ax.set_axisbelow(True)
        filled  = ax.scatter([], [], color='gray', s=80, label='GT (bbox)')
        outline = ax.scatter([], [], facecolors='white', edgecolors='gray',
                             s=80, linewidths=2.2, label='YOLO crop')
        ax.legend(handles=[filled, outline], loc='lower right', framealpha=0.9)
    fig.suptitle(f'Phase 2 — {_ds_title(dataset)}: Accuracy Drop  GT -> YOLO',
                 fontweight='bold')
    fig.tight_layout()
    _save_fig(fig, f'phase2_dumbbell_{dataset}')


def _plot_fold_boxplot(state, dataset):
    np.random.seed(0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, mk, label in zip(axes, _MODEL_KEYS, _MODEL_LABELS):
        gt_vals, yolo_vals = state[mk]['gt']['top1'], state[mk]['yolo']['top1']
        color = _COLORS[label]
        bp = ax.boxplot(
            [gt_vals, yolo_vals], patch_artist=True, widths=0.44,
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
            flierprops=dict(marker=''),
        )
        bp['boxes'][0].set_facecolor(color); bp['boxes'][0].set_alpha(0.65)
        bp['boxes'][1].set_facecolor(color); bp['boxes'][1].set_alpha(0.30)
        for patch in bp['boxes']:
            patch.set_linewidth(1.2)
        for gt, yolo in zip(gt_vals, yolo_vals):
            ax.plot([1, 2], [gt, yolo], color='gray', alpha=0.3, lw=0.9, zorder=1)
        for vals, x_pos in [(gt_vals, 1), (yolo_vals, 2)]:
            jitter = np.random.normal(x_pos, 0.06, len(vals))
            ax.scatter(jitter, vals, s=20, alpha=0.7, color=color,
                       edgecolors='white', linewidths=0.4, zorder=3)
        ax.set_xticks([1, 2]); ax.set_xticklabels(['GT', 'YOLO'])
        ax.set_title(label, fontsize=9); ax.set_ylabel('Top-1 Accuracy')
        ax.yaxis.grid(True, linestyle=':', alpha=0.5); ax.set_axisbelow(True)
    fig.suptitle(f'Phase 2 — {_ds_title(dataset)}: Per-Fold GT vs YOLO Top-1',
                 fontweight='bold')
    fig.tight_layout()
    _save_fig(fig, f'phase2_fold_boxplot_{dataset}')


def generate_phase2_plots(state, dataset):
    print(f'\nGenerating Phase 2 figures for {dataset}...')
    _plot_bars(state, dataset)
    _plot_dumbbell(state, dataset)
    _plot_fold_boxplot(state, dataset)


# ── CLI & main ───────────────────────────────────────────────────────────────

def _phase2_complete(state_file, total_folds):
    if not os.path.exists(state_file):
        return False
    with open(state_file) as fh:
        state = json.load(fh)
    for mk in _MODEL_KEYS:
        r = state.get(mk, {})
        if len(r.get('gt', {}).get('top1', [])) < total_folds:
            return False
    return True


def get_config():
    parser = argparse.ArgumentParser(description="Phase 2: GT vs YOLO test-set evaluation")
    parser.add_argument('--dataset',       type=str, required=True,
                        choices=['stanford', 'compcars'])
    parser.add_argument('--batch_size',    type=int,   default=16)
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--folds',         type=int,   default=5)
    parser.add_argument('--repeats',       type=int,   default=2)
    parser.add_argument('--lr_new',        type=float, default=1e-3)
    parser.add_argument('--lr_base',       type=float, default=1e-5)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int,   default=5)
    parser.add_argument('--seed',          type=int,   default=21)
    parser.add_argument('--yolo_model',    type=str,   default='yolo26s.pt')
    args = parser.parse_args()

    if args.dataset == 'stanford':
        args.train_csv = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\StanfordCars_devkit\cars_train.csv"
        args.test_csv  = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\StanfordCars_devkit\cars_test.csv"
        args.train_dir = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\StanfordCars\cars_train\cars_train"
        args.test_dir  = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\StanfordCars\cars_test\cars_test"
    else:
        args.train_csv = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\CompCars_devkit\compcars_dataset.csv"
        args.test_csv  = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\CompCars_devkit\compcars_dataset.csv"
        args.train_dir = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\image"
        args.test_dir  = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\image"
        args.train_txt = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\train_test_split\classification\train.txt"
        args.test_txt  = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\train_test_split\classification\test.txt"

    args.results_file = f'phase2_results_{args.dataset}.txt'
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

    state_file  = f'phase2_state_{args.dataset}.json'
    total_folds = args.folds * args.repeats

    if _phase2_complete(state_file, total_folds):
        print(f"All Phase 2 results complete for {args.dataset} — generating figures (no training).")
        for ds in ['stanford', 'compcars']:
            sf = f'phase2_state_{ds}.json'
            if _phase2_complete(sf, total_folds):
                with open(sf) as fh:
                    state = json.load(fh)
                generate_phase2_plots(state, ds)
        print(f"\nDone. Figures saved to {_FIG_DIR}")
        return

    print(f"Device:  {args.device}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  Folds: {args.folds}x{args.repeats}"
          f"  |  Seed: {args.seed}")
    print(f"LR base: {args.lr_base}  |  LR new: {args.lr_new}  |  WD: {args.weight_decay}"
          f"  |  Warmup: {args.warmup_epochs}\n")

    train_df = pd.read_csv(args.train_csv)
    if args.dataset == 'compcars':
        with open(args.train_txt) as f:
            train_files = {l.strip() for l in f}
        train_df = train_df[train_df['filename'].isin(train_files)].reset_index(drop=True)
        print(f"Training images: {len(train_df)}")

    le = LabelEncoder()
    train_df['mapped_label'] = le.fit_transform(train_df['class_id'])
    num_classes = len(le.classes_)
    print(f"Classes: {num_classes}")

    test_df = pd.read_csv(args.test_csv)
    if args.dataset == 'compcars':
        with open(args.test_txt) as f:
            test_files = {l.strip() for l in f}
        test_df = test_df[test_df['filename'].isin(test_files)].reset_index(drop=True)
        print(f"Test images:     {len(test_df)}")
    test_df['mapped_label'] = le.transform(test_df['class_id'])

    gt_test_loader = DataLoader(
        CarsDataset(test_df, args.test_dir, transform=val_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    yolo_pil, yolo_lbl = load_or_build_yolo_dataset(
        test_df, args.test_dir, args.yolo_model, f'yolo_cache_{args.dataset}.json'
    )
    yolo_test_loader = DataLoader(
        YOLODataset(yolo_pil, yolo_lbl, transform=val_transform),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    train_base = CarsDataset(train_df, args.train_dir, transform=train_transform)
    targets    = train_df['mapped_label'].values
    rskf  = RepeatedStratifiedKFold(
        n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed
    )
    folds = list(rskf.split(np.zeros(len(targets)), targets))

    model_configs = [
        ('resnet', 'Standard ResNet50', create_standard_resnet),
        ('se',     'SE ResNet50',       create_se_resnet),
        ('cbam',   'CBAM ResNet50',     create_cbam_resnet),
    ]

    if os.path.exists(state_file):
        with open(state_file) as f:
            all_results = json.load(f)
        for mk, _, _ in model_configs:
            if mk in all_results:
                n = len(all_results[mk]['gt']['top1'])
                status = 'complete' if n >= total_folds else f'{n}/{total_folds} folds done'
                print(f"  Loaded {mk}: {status}")
        print()
    else:
        all_results = {}

    def save_state(model_key, partial_results):
        all_results[model_key] = partial_results
        with open(state_file, 'w') as f:
            json.dump(all_results, f)

    for model_key, model_label, create_fn in model_configs:
        existing = all_results.get(model_key, {
            'gt':   {'top1': [], 'top5': [], 'f1': []},
            'yolo': {'top1': [], 'top5': [], 'f1': []},
        })
        if len(existing['gt']['top1']) >= total_folds:
            print(f"\nSkipping {model_label} — all {total_folds} folds complete.")
            continue

        print(f"\n{'='*55}\nModel: {model_label}\n{'='*55}")
        all_results[model_key] = run_model_folds(
            create_fn, num_classes, train_base, folds,
            gt_test_loader, yolo_test_loader, args,
            existing_results=existing,
            on_fold_complete=lambda r, k=model_key: save_state(k, r),
        )

    stats_gt_top1 = run_stats_3models(
        all_results['resnet']['gt']['top1'],
        all_results['se']['gt']['top1'],
        all_results['cbam']['gt']['top1'],
        "Top-1 Accuracy", "Experiment A — Ground Truth"
    )
    stats_gt_f1 = run_stats_3models(
        all_results['resnet']['gt']['f1'],
        all_results['se']['gt']['f1'],
        all_results['cbam']['gt']['f1'],
        "Macro F1", "Experiment A — Ground Truth"
    )
    stats_yolo_top1 = run_stats_3models(
        all_results['resnet']['yolo']['top1'],
        all_results['se']['yolo']['top1'],
        all_results['cbam']['yolo']['top1'],
        "Top-1 Accuracy", "Experiment B — YOLO"
    )
    stats_yolo_f1 = run_stats_3models(
        all_results['resnet']['yolo']['f1'],
        all_results['se']['yolo']['f1'],
        all_results['cbam']['yolo']['f1'],
        "Macro F1", "Experiment B — YOLO"
    )

    degradation_stats = []
    for model_key, model_label, _ in model_configs:
        for metric in ['top1', 'f1']:
            degradation_stats.append(run_stats_gt_vs_yolo(
                all_results[model_key]['gt'][metric],
                all_results[model_key]['yolo'][metric],
                model_label, metric
            ))

    with open(args.results_file, 'w') as f:
        f.write(f"=== PHASE 2 RESULTS: {args.dataset.upper()} ===\n")
        f.write(f"Config: {args.folds}-Fold x{args.repeats} repeats, {args.epochs} epochs,"
                f" seed={args.seed}\n")
        f.write(f"LR base: {args.lr_base}  LR new: {args.lr_new}  WD: {args.weight_decay}"
                f"  Warmup: {args.warmup_epochs}\n")
        f.write(f"YOLO model: {args.yolo_model}\n\n")

        for model_key, model_label, _ in model_configs:
            f.write(f"\n{'='*55}\n{model_label}\n{'='*55}\n")
            gt   = all_results[model_key]['gt']
            yolo = all_results[model_key]['yolo']
            for i in range(total_folds):
                f.write(f"Fold {i+1}:\n")
                f.write(f"  GT:   Top-1={gt['top1'][i]:.4f}  Top-5={gt['top5'][i]:.4f}"
                        f"  F1={gt['f1'][i]:.4f}\n")
                f.write(f"  YOLO: Top-1={yolo['top1'][i]:.4f}  Top-5={yolo['top5'][i]:.4f}"
                        f"  F1={yolo['f1'][i]:.4f}\n")
                f.write(f"  Drop: Top-1={gt['top1'][i]-yolo['top1'][i]:+.4f}  "
                        f"Top-5={gt['top5'][i]-yolo['top5'][i]:+.4f}  "
                        f"F1={gt['f1'][i]-yolo['f1'][i]:+.4f}\n")

        f.write(f"\n{'='*55}\nAVERAGE SCORES\n{'='*55}\n")
        f.write(f"{'Model':<22} {'Cond':<6} {'Top-1':>8} {'Top-5':>8} {'Macro-F1':>10}\n")
        f.write("-" * 57 + "\n")
        for model_key, model_label, _ in model_configs:
            for cond, label in [('gt', 'GT'), ('yolo', 'YOLO')]:
                r = all_results[model_key][cond]
                f.write(f"{model_label:<22} {label:<6} "
                        f"{np.mean(r['top1']):>8.4f} {np.mean(r['top5']):>8.4f}"
                        f" {np.mean(r['f1']):>10.4f}\n")

        f.write(f"\n\n{'='*55}\nEXPERIMENT A — GROUND TRUTH: 3-MODEL COMPARISON\n{'='*55}\n")
        f.write(stats_gt_top1 + "\n" + stats_gt_f1 + "\n")
        f.write(f"\n\n{'='*55}\nEXPERIMENT B — YOLO: 3-MODEL COMPARISON\n{'='*55}\n")
        f.write(stats_yolo_top1 + "\n" + stats_yolo_f1 + "\n")
        f.write(f"\n\n{'='*55}\nACCURACY DEGRADATION: GT vs YOLO (per model)\n{'='*55}\n")
        for s in degradation_stats:
            f.write(s + "\n")

    print(f"\n{'='*57}")
    print(f"{'Model':<22} {'Cond':<6} {'Top-1':>8} {'Top-5':>8} {'Macro-F1':>10}")
    print("-" * 57)
    for model_key, model_label, _ in model_configs:
        for cond, label in [('gt', 'GT'), ('yolo', 'YOLO')]:
            r = all_results[model_key][cond]
            print(f"{model_label:<22} {label:<6} "
                  f"{np.mean(r['top1']):>8.4f} {np.mean(r['top5']):>8.4f}"
                  f" {np.mean(r['f1']):>10.4f}")
    print(f"{'='*57}")
    print(stats_gt_top1); print(stats_gt_f1)
    print(stats_yolo_top1); print(stats_yolo_f1)
    for s in degradation_stats:
        print(s)

    print(f"\nResults saved to {args.results_file}")

    generate_phase2_plots(all_results, args.dataset)


if __name__ == '__main__':
    main()
