import argparse
import json
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

from models import create_standard_resnet, create_se_resnet, create_cbam_resnet
from datasets import CarsDataset, train_transform, val_transform


# GradCAM

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = [
            model.layer4[-1].register_forward_hook(self._fwd_hook),
            model.layer4[-1].register_full_backward_hook(self._bwd_hook),
        ]

    def _fwd_hook(self, _module, _input, output):
        self.activations = output.detach()

    def _bwd_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, pred_class].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().float().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, pred_class

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def _overlay(img_np, cam_np, alpha=0.45):
    H, W = img_np.shape[:2]
    cam_t = torch.tensor(cam_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    cam_up = torch.nn.functional.interpolate(cam_t, size=(H, W), mode='bilinear', align_corners=False)
    heatmap = plt.cm.jet(cam_up.squeeze().numpy())[:, :, :3]
    return np.clip(alpha * heatmap + (1 - alpha) * img_np, 0, 1)


# Coordinate descent search space

BASE_CONFIG = {
    'epochs':        20,
    'batch_size':    32,
    'lr_new':        1e-3,
    'lr_base':       1e-5,
    'warmup_epochs': 3,
    'weight_decay':  1e-4,
}

SEARCH_PHASES = [
    ('epochs',        [10, 20, 30, 50]),
    ('batch_size',    [16, 32, 64]),
    ('warmup_epochs', [0, 3, 5, 10]),
    ('lr_new',        [1e-2, 1e-3, 1e-4]),
]

MODEL_FACTORIES = [
    ('Standard ResNet50', create_standard_resnet),
    ('SE ResNet50',       create_se_resnet),
    ('CBAM ResNet50',     create_cbam_resnet),
]


def config_tag(cfg):
    return (f"ep{cfg['epochs']}_bs{cfg['batch_size']}_w{cfg['warmup_epochs']}"
            f"_lrn{cfg['lr_new']:.0e}_lrb{cfg['lr_base']:.0e}_wd{cfg['weight_decay']:.0e}")


def make_loaders(df, img_dir, train_idx, val_idx, batch_size):
    train_ds = CarsDataset(df, img_dir, transform=train_transform)
    val_ds   = CarsDataset(df, img_dir, transform=val_transform)
    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(Subset(val_ds,   val_idx),   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, val_ds


def train_one_model(model, cfg, train_loader, val_loader, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    new_params, base_params = [], []
    for name, param in model.named_parameters():
        if 'fc' in name or 'se' in name or 'cbam' in name:
            new_params.append(param)
        else:
            base_params.append(param)

    optimizer = optim.Adam([
        {'params': base_params, 'lr': cfg['lr_base']},
        {'params': new_params,  'lr': cfg['lr_new']},
    ], weight_decay=cfg['weight_decay'])

    epochs = cfg['epochs']
    warmup = cfg['warmup_epochs']

    if warmup > 0:
        sch_w = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)
        sch_c = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sch_w, sch_c], milestones=[warmup])
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
            top1_correct += correct[:1].reshape(-1).float().sum().item()
            top5_correct += correct[:5].reshape(-1).float().sum().item()
            total += labels.size(0)
            all_preds.extend(preds[0].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        'top1': top1_correct / total,
        'top5': top5_correct / total,
        'f1':   f1_score(all_labels, all_preds, average='macro'),
    }
    return metrics, model


def save_gradcam(trained_models, val_ds, val_idx, device, le, tag, num_images=2):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def denorm(t):
        return np.clip(t.cpu().numpy().transpose(1, 2, 0) * std + mean, 0, 1)

    chosen = random.sample(list(val_idx), min(num_images, len(val_idx)))

    for model in trained_models.values():
        model.to(device).eval()

    ncols = len(trained_models) + 1

    for img_idx, sample_idx in enumerate(chosen):
        img_tensor, true_label = val_ds[sample_idx]
        orig = denorm(img_tensor)
        true_class = le.inverse_transform([true_label])[0]

        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        fig.suptitle(f"GradCAM | {tag}\nTrue: {true_class}", fontsize=9, y=1.02)

        axes[0].imshow(orig)
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis('off')

        for ax_idx, (name, model) in enumerate(trained_models.items(), 1):
            gc = GradCAM(model)
            cam, pred_idx = gc.generate(img_tensor.unsqueeze(0).to(device))
            gc.remove_hooks()

            pred_class = le.inverse_transform([pred_idx])[0]
            correct = (pred_idx == true_label)
            axes[ax_idx].imshow(_overlay(orig, cam))
            axes[ax_idx].set_title(
                f"{name}\nPred: {pred_class}  ({'OK' if correct else 'WRONG'})",
                fontsize=8,
                color='green' if correct else 'red'
            )
            axes[ax_idx].axis('off')

        plt.tight_layout()
        fname = f"gradcam_{tag}_img{img_idx + 1}.png"
        plt.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"    Saved {fname}")

    for model in trained_models.values():
        model.cpu()
    torch.cuda.empty_cache()


def get_config():
    parser = argparse.ArgumentParser(description="Coordinate descent hyperparameter search — all 3 models")
    parser.add_argument('--dataset', type=str, required=True, choices=['stanford', 'compcars'])
    args = parser.parse_args()

    if args.dataset == 'stanford':
        args.csv_path = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\StanfordCars_devkit\cars_train.csv"
        args.img_dir  = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\StanfordCars\cars_train\cars_train"
    elif args.dataset == 'compcars':
        args.csv_path = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\CompCars_devkit\compcars_dataset.csv"
        args.img_dir  = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\image"
        args.compcars_train_txt = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\train_test_split\classification\train.txt"

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def set_seed(seed=21):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_results(all_runs, best_config, dataset):
    # Global y range so all panels are directly comparable
    all_top1 = [r['metrics'][n]['top1'] for r in all_runs for n, _ in MODEL_FACTORIES]
    y_min = min(all_top1)
    y_max = max(all_top1)
    y_pad = max((y_max - y_min) * 0.18, 0.025)
    ylim = (max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))

    # Final config score: avg Top-1 of the best run in the last search phase
    last_phase = SEARCH_PHASES[-1][0]
    final_run = next(
        r for r in reversed(all_runs)
        if r['phase'] == last_phase and r['param_val'] == best_config[last_phase]
    )
    final_score = np.mean([final_run['metrics'][n]['top1'] for n, _ in MODEL_FACTORIES])

    LOG_SCALE_PARAMS = {'lr_new'}

    colors  = {'Standard ResNet50': '#4C72B0', 'SE ResNet50': '#DD8452', 'CBAM ResNet50': '#55A868'}
    lmarkers = {'Standard ResNet50': 'o', 'SE ResNet50': 's', 'CBAM ResNet50': '^'}

    n_phases = len(SEARCH_PHASES)
    ncols = 2
    nrows = (n_phases + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.8 * nrows))
    axes = axes.flatten()

    for ax_idx, (param_name, candidates) in enumerate(SEARCH_PHASES):
        ax = axes[ax_idx]
        phase_runs = [r for r in all_runs if r['phase'] == param_name]
        use_log = param_name in LOG_SCALE_PARAMS

        # x positions
        x_pos    = candidates if use_log else list(range(len(candidates)))
        x_labels = [f'{v:.0e}' for v in candidates] if use_log else [str(v) for v in candidates]
        best_idx = candidates.index(best_config[param_name])
        best_x   = candidates[best_idx] if use_log else best_idx

        # Per-model lines (dashed + dots)
        for model_name, _ in MODEL_FACTORIES:
            y = [r['metrics'][model_name]['top1'] for r in phase_runs]
            ax.plot(x_pos, y, marker=lmarkers[model_name], color=colors[model_name],
                    label=model_name, linewidth=1.5, markersize=7, linestyle='--', zorder=2)

        # Average line (dotted to distinguish from model lines)
        avg_y = [np.mean([r['metrics'][n]['top1'] for n, _ in MODEL_FACTORIES])
                 for r in phase_runs]
        ax.plot(x_pos, avg_y, marker='D', color='black', linestyle=':',
                label='Average', linewidth=1.5, markersize=6, alpha=0.8, zorder=2)

        # Chosen value: faint vertical line + filled star on the average curve
        ax.axvline(x=best_x, color='crimson', linestyle='--', linewidth=1.2,
                   alpha=0.35, zorder=1)
        ax.scatter([best_x], [avg_y[best_idx]], marker='*', s=320, color='crimson',
                   zorder=5, label=f'Chosen: {best_config[param_name]}')

        # Horizontal reference at final config score
        ax.axhline(y=final_score, color='dimgray', linestyle=':', linewidth=1.2,
                   alpha=0.6, zorder=1, label=f'Final config: {final_score:.4f}')

        if use_log:
            ax.set_xscale('log')
            ax.set_xticks(candidates)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0e}'))
        else:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=9)

        ax.set_title(param_name, fontweight='bold', fontsize=11)
        ax.set_ylabel('Top-1 Accuracy', fontsize=9)
        ax.set_ylim(ylim)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        ax.legend(fontsize=7.5, loc='lower right')

    for ax_idx in range(n_phases, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        f"Coordinate Descent Pretest  -  {dataset.upper()}\n"
        f"Fixed: lr_base=1e-05, weight_decay=1e-04",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    out = f"pretest_chart_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to {out}")


def save_run_data(all_runs, best_config, dataset):
    data = {
        'dataset':     dataset,
        'best_config': best_config,
        'all_runs': [
            {
                'run':       r['run'],
                'phase':     r['phase'],
                'param_val': r['param_val'],
                'config':    r['config'],
                'metrics': {
                    model_name: {k: float(v) for k, v in m.items()}
                    for model_name, m in r['metrics'].items()
                },
            }
            for r in all_runs
        ],
    }
    out = f"pretest_data_{dataset}.json"
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Run data saved to {out}")


def write_results(all_runs, best_config, dataset):
    lines = []
    lines.append("=" * 70)
    lines.append(f"COORDINATE DESCENT PRETEST RESULTS  -  {dataset.upper()}")
    lines.append("Best config at each phase selected by average Top-1 across all 3 models.")
    lines.append("=" * 70)

    for param_name, candidates in SEARCH_PHASES:
        phase_runs = [r for r in all_runs if r['phase'] == param_name]
        lines.append(f"\n--- Phase: {param_name}  (candidates: {candidates}) ---")
        header = f"  {'Value':<12} {'Std Top-1':>10} {'SE Top-1':>10} {'CBAM Top-1':>11} {'Avg Top-1':>10}"
        lines.append(header)
        lines.append("  " + "-" * 57)
        for r in phase_runs:
            avg = np.mean([r['metrics'][n]['top1'] for n, _ in MODEL_FACTORIES])
            marker = " <-- best" if r['param_val'] == best_config[param_name] else ""
            lines.append(
                f"  {str(r['param_val']):<12}"
                f" {r['metrics']['Standard ResNet50']['top1']:>10.4f}"
                f" {r['metrics']['SE ResNet50']['top1']:>10.4f}"
                f" {r['metrics']['CBAM ResNet50']['top1']:>11.4f}"
                f" {avg:>10.4f}{marker}"
            )

    lines.append("\n" + "=" * 70)
    lines.append("FINAL BEST CONFIGURATION")
    lines.append("=" * 70)
    for k, v in best_config.items():
        lines.append(f"  {k:<20}: {v}")

    lines.append("\n" + "=" * 70)
    lines.append("ALL RUNS (chronological)")
    lines.append("=" * 70)
    for r in all_runs:
        avg = np.mean([r['metrics'][n]['top1'] for n, _ in MODEL_FACTORIES])
        lines.append(f"\nRun {r['run']:>2}  phase={r['phase']}  {r['phase']}={r['param_val']}  avg_top1={avg:.4f}")
        lines.append(f"  Config: {config_tag(r['config'])}")
        for model_name, _ in MODEL_FACTORIES:
            m = r['metrics'][model_name]
            lines.append(f"  {model_name:<22} Top-1={m['top1']:.4f}  Top-5={m['top5']:.4f}  F1={m['f1']:.4f}")

    text = "\n".join(lines)
    print("\n" + text)

    out_file = f"pretest_results_{dataset}.txt"
    with open(out_file, 'w') as f:
        f.write(text)
    print(f"\nResults saved to {out_file}")


def main():
    args = get_config()
    set_seed(42)

    data_file = f"pretest_data_{args.dataset}.json"
    if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
        print(f"Found existing results in {data_file} — skipping training, regenerating plot.")
        with open(data_file) as f:
            data = json.load(f)
        plot_results(data['all_runs'], data['best_config'], args.dataset)
        return

    total_runs = sum(len(cands) for _, cands in SEARCH_PHASES)
    print(f"Device:     {args.device}")
    print(f"Dataset:    {args.dataset.upper()}")
    print(f"Search:     {total_runs} configs x 3 models = {total_runs * 3} training runs total")
    print(f"Base config: {BASE_CONFIG}\n")

    df = pd.read_csv(args.csv_path)
    if args.dataset == 'compcars':
        with open(args.compcars_train_txt, 'r') as f:
            valid = {l.strip() for l in f}
        df = df[df['filename'].isin(valid)].reset_index(drop=True)
        print(f"Filtered CompCars to {len(df)} training images.")

    le = LabelEncoder()
    df['mapped_label'] = le.fit_transform(df['class_id'])
    num_classes = len(le.classes_)
    targets = df['mapped_label'].values
    print(f"Classes: {num_classes}\n")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    print(f"Train: {len(train_idx)}  |  Val: {len(val_idx)}\n")

    current  = dict(BASE_CONFIG)
    all_runs = []
    run_num  = 0

    for param_name, candidates in SEARCH_PHASES:
        print(f"\n{'='*62}")
        print(f"Phase: tuning [{param_name}]  candidates: {candidates}")
        print(f"Fixed: { {k: v for k, v in current.items() if k != param_name} }")
        print(f"{'='*62}")

        phase_results = []

        for val in candidates:
            run_num += 1
            cfg = dict(current)
            cfg[param_name] = val
            tag = config_tag(cfg)

            print(f"\nConfig {run_num}/{total_runs}  |  {param_name}={val}")
            print(f"  Tag: {tag}")

            train_loader, val_loader, val_ds = make_loaders(
                df, args.img_dir, train_idx, val_idx, cfg['batch_size']
            )

            run_metrics = {}
            trained_models = {}

            for model_name, factory in MODEL_FACTORIES:
                print(f"  Training {model_name}...", end='', flush=True)
                metrics, model = train_one_model(
                    factory(num_classes), cfg, train_loader, val_loader, args.device
                )
                run_metrics[model_name]   = metrics
                trained_models[model_name] = model
                print(f"  Top-1={metrics['top1']:.4f}  Top-5={metrics['top5']:.4f}  F1={metrics['f1']:.4f}")

            avg_top1 = np.mean([run_metrics[n]['top1'] for n, _ in MODEL_FACTORIES])
            print(f"  Avg Top-1: {avg_top1:.4f}")

            print("  GradCAM...")
            save_gradcam(trained_models, val_ds, val_idx, args.device, le, tag)

            for m in trained_models.values():
                del m
            torch.cuda.empty_cache()

            phase_results.append((val, avg_top1))
            all_runs.append({
                'run':       run_num,
                'phase':     param_name,
                'param_val': val,
                'config':    cfg,
                'metrics':   run_metrics,
            })

        best_val = max(phase_results, key=lambda x: x[1])[0]
        best_avg = max(phase_results, key=lambda x: x[1])[1]
        current[param_name] = best_val
        print(f"\n  -> Best {param_name} = {best_val}  (avg Top-1: {best_avg:.4f})")

    save_run_data(all_runs, current, args.dataset)
    write_results(all_runs, current, args.dataset)
    plot_results(all_runs, current, args.dataset)


if __name__ == '__main__':
    main()
