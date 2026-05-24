import argparse
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


def visualize_gradcam(trained_models, val_loader, device, le, num_images=2):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    def denorm(t):
        return np.clip(t.cpu().numpy().transpose(1, 2, 0) * std + mean, 0, 1)

    # Collect samples from the (unshuffled) val loader
    samples = []
    for imgs, labels in val_loader:
        for i in range(imgs.size(0)):
            samples.append((imgs[i], labels[i].item()))
            if len(samples) >= num_images:
                break
        if len(samples) >= num_images:
            break

    ncols = len(trained_models) + 1

    for img_idx, (img_tensor, true_label) in enumerate(samples):
        orig = denorm(img_tensor)
        true_class = le.inverse_transform([true_label])[0]

        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        fig.suptitle(
            f"GradCAM  —  Image {img_idx + 1}  |  True class: {true_class}",
            fontsize=13, fontweight='bold', y=1.03
        )

        axes[0].imshow(orig)
        axes[0].set_title("Original", fontsize=11)
        axes[0].axis('off')

        for ax_idx, (name, model) in enumerate(trained_models.items(), 1):
            model = model.to(device)
            gc = GradCAM(model)
            cam, pred_idx = gc.generate(img_tensor.unsqueeze(0).to(device))
            gc.remove_hooks()
            model.cpu()
            torch.cuda.empty_cache()

            pred_class = le.inverse_transform([pred_idx])[0]
            correct = (pred_idx == true_label)
            axes[ax_idx].imshow(_overlay(orig, cam))
            axes[ax_idx].set_title(
                f"{name}\nPred: {pred_class}  {'✓' if correct else '✗'}",
                fontsize=10,
                color='green' if correct else 'red'
            )
            axes[ax_idx].axis('off')

        plt.tight_layout()
        out = f'gradcam_{img_idx + 1}.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out}")


def get_config():
    parser = argparse.ArgumentParser(description="Hyperparameter pretest — single stratified split")
    parser.add_argument('--dataset',    type=str, required=True, choices=['stanford', 'compcars'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--lr_new',     type=float, default=1e-3,  help="LR for attention modules + FC head")
    parser.add_argument('--lr_base',    type=float, default=1e-5,  help="LR for pretrained backbone")
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int,   default=5,
                        help="Epochs to linearly ramp LR from 10%% of target to full target before cosine decay")
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = get_config()
    set_seed(42)

    print(f"Device:     {args.device}")
    print(f"Dataset:    {args.dataset.upper()}")
    print(f"Batch size: {args.batch_size}  |  Epochs: {args.epochs}")
    print(f"LR base:    {args.lr_base}     |  LR new: {args.lr_new}  |  WD: {args.weight_decay}  |  Warmup: {args.warmup_epochs} epochs\n")

    df = pd.read_csv(args.csv_path)

    if args.dataset == 'compcars':
        with open(args.compcars_train_txt, 'r') as f:
            valid_filenames = [line.strip() for line in f.readlines()]
        df = df[df['filename'].isin(valid_filenames)].reset_index(drop=True)
        print(f"Filtered CompCars down to {len(df)} official training images.")

    le = LabelEncoder()
    df['mapped_label'] = le.fit_transform(df['class_id'])
    num_classes = len(le.classes_)
    print(f"Classes: {num_classes}\n")

    targets = df['mapped_label'].values

    # Single stratified 80/20 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_dataset = CarsDataset(df, args.img_dir, transform=train_transform)
    val_dataset   = CarsDataset(df, args.img_dir, transform=val_transform)

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(Subset(val_dataset,   val_idx),   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_idx)}  |  Val samples: {len(val_idx)}\n")

    model_configs = [
        ('Standard ResNet50', lambda: create_standard_resnet(num_classes)),
        ('SE ResNet50',       lambda: create_se_resnet(num_classes)),
        ('CBAM ResNet50',     lambda: create_cbam_resnet(num_classes)),
    ]

    def train_with_args(model):
        model = model.to(args.device)
        criterion = nn.CrossEntropyLoss()

        new_params, base_params = [], []
        for name, param in model.named_parameters():
            if 'fc' in name or 'se' in name or 'cbam' in name:
                new_params.append(param)
            else:
                base_params.append(param)

        optimizer = optim.Adam([
            {'params': base_params, 'lr': args.lr_base},
            {'params': new_params,  'lr': args.lr_new},
        ], weight_decay=args.weight_decay)

        if args.warmup_epochs > 0:
            # Linear ramp: 10% of target LR → full LR over warmup_epochs
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - args.warmup_epochs)
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(args.epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
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
                inputs, labels = inputs.to(args.device), labels.to(args.device)
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
        model.cpu()
        torch.cuda.empty_cache()
        return model, metrics

    results = {}
    trained_models = {}
    for label, create_fn in model_configs:
        print(f"Training {label}...")
        model, metrics = train_with_args(create_fn())
        results[label] = metrics
        trained_models[label] = model
        m = metrics
        print(f"  Top-1: {m['top1']:.4f}  |  Top-5: {m['top5']:.4f}  |  Macro-F1: {m['f1']:.4f}\n")

    print("=" * 55)
    print(f"{'Model':<22} {'Top-1':>7} {'Top-5':>7} {'Macro-F1':>10}")
    print("-" * 55)
    for label, m in results.items():
        print(f"{label:<22} {m['top1']:>7.4f} {m['top5']:>7.4f} {m['f1']:>10.4f}")
    print("=" * 55)

    print("\nGenerating GradCAM visualizations...")
    visualize_gradcam(trained_models, val_loader, args.device, le)

    for m in trained_models.values():
        del m
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
