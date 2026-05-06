import argparse
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

# Training logic & metrics

def train_and_evaluate(model, train_loader, val_loader, device, epochs):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    new_params = []
    base_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name or 'se' in name or 'cbam' in name:
            new_params.append(param)
        else:
            base_params.append(param)
            
    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-5},
        {'params': new_params, 'lr': 1e-4}   
    ], weight_decay=1e-4)
    
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
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

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
        'f1': f1_score(all_labels, all_preds, average='macro')
    }
    
    del model
    torch.cuda.empty_cache()
    return metrics

# Statistical analysis

def run_statistics(res_std, res_se, res_cbam, metric_name):
    output = []
    output.append(f"\n--- Statistical Analysis ({metric_name}) ---")
    
    # Friedman Test
    stat, p_friedman = friedmanchisquare(res_std, res_se, res_cbam)
    output.append(f"Friedman Test: Statistic={stat:.4f}, p-value={p_friedman:.4e}")
    
    if p_friedman < 0.05:
        output.append("Result: Significant differences exist among the models. Running Wilcoxon Post-hoc...")
        
        # Wilcoxon Signed-Rank Test (with Bonferroni Correction for 3 comparisons: alpha = 0.0167)
        alpha_corrected = 0.05 / 3
        output.append(f"Bonferroni-corrected significance threshold: alpha = {alpha_corrected:.4f}")
        
        def safe_wilcoxon(x, y):
            # Wilcoxon fails if all differences are perfectly zero
            if np.allclose(np.array(x) - np.array(y), 0):
                return 0.0, 1.0 
            return wilcoxon(x, y)

        stat_1, p_1 = safe_wilcoxon(res_std, res_se)
        stat_2, p_2 = safe_wilcoxon(res_std, res_cbam)
        stat_3, p_3 = safe_wilcoxon(res_se, res_cbam)
        
        output.append(f"  Std vs SE:   p-value = {p_1:.4e} {'(Significant)' if p_1 < alpha_corrected else '(Not Sig)'}")
        output.append(f"  Std vs CBAM: p-value = {p_2:.4e} {'(Significant)' if p_2 < alpha_corrected else '(Not Sig)'}")
        output.append(f"  SE vs CBAM:  p-value = {p_3:.4e} {'(Significant)' if p_3 < alpha_corrected else '(Not Sig)'}")
    else:
        output.append("Result: No statistically significant differences found among the models (p >= 0.05).")
        
    return "\n".join(output)

# Config & main pipeline

def get_config():
    parser = argparse.ArgumentParser(description="Vehicle Recognition Experiment")
    parser.add_argument('--dataset', type=str, required=True, choices=['stanford', 'compcars'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()

    if args.dataset == 'stanford':
        args.csv_path = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\StanfordCars_devkit\cars_train.csv"
        args.img_dir = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\StanfordCars\cars_train\cars_train"
        args.num_classes = 196
    elif args.dataset == 'compcars':
        args.csv_path = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\CompCars_devkit\compcars_dataset.csv"
        args.img_dir = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\image"
        args.num_classes = 431 
        args.compcars_train_txt = r"c:\Users\pc\Desktop\pracamag\VehicleRecognition\data\CompCars\train_test_split\classification\train.txt"

    args.results_file = f'experiment_results_{args.dataset}.txt'
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
    
    print(f"Device: {args.device}")
    print(f"Dataset: {args.dataset.upper()}")
    
    df = pd.read_csv(args.csv_path)
    
    if args.dataset == 'compcars':
    # Read the text file into a list of strings, stripping out newline characters
        with open(args.compcars_train_txt, 'r') as f:
            valid_filenames = [line.strip() for line in f.readlines()]
        df = df[df['filename'].isin(valid_filenames)].reset_index(drop=True)
    
        print(f"Filtered CompCars down to {len(df)} official training images.")

    le = LabelEncoder()
    df['mapped_label'] = le.fit_transform(df['class_id'])
    
    args.num_classes = len(le.classes_)
    print(f"Detected {args.num_classes} unique classes.")
    
    print(f"Starting {args.folds}-Fold RSKF (Repeats: {args.repeats}, Epochs: {args.epochs})\n")
    
    train_dataset_base = CarsDataset(df, args.img_dir, transform=train_transform)
    val_dataset_base = CarsDataset(df, args.img_dir, transform=val_transform)
    
    # Extract the safe targets for stratification
    targets = train_dataset_base.df['mapped_label'].values
    
    rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=42)
    
    results = {
        'resnet': {'top1': [], 'top5': [], 'f1': []},
        'se':     {'top1': [], 'top5': [], 'f1': []},
        'cbam':   {'top1': [], 'top5': [], 'f1': []}
    }

    for fold, (train_idx, val_idx) in enumerate(rskf.split(np.zeros(len(targets)), targets)):
        print(f"--- Iteration {fold + 1}/{args.folds * args.repeats} ---")
        
        train_sub = Subset(train_dataset_base, train_idx)
        val_sub = Subset(val_dataset_base, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        # ResNet50
        print("Training Standard ResNet50...")
        metrics_resnet = train_and_evaluate(create_standard_resnet(args.num_classes), train_loader, val_loader, args.device, args.epochs)
        for k in results['resnet']: results['resnet'][k].append(metrics_resnet[k])
        
        # ResNet50 + SE
        print("Training SE ResNet50...")
        metrics_se = train_and_evaluate(create_se_resnet(args.num_classes), train_loader, val_loader, args.device, args.epochs)
        for k in results['se']: results['se'][k].append(metrics_se[k])

        # ResNet50 + CBAM
        print("Training CBAM ResNet50...")
        metrics_cbam = train_and_evaluate(create_cbam_resnet(args.num_classes), train_loader, val_loader, args.device, args.epochs)
        for k in results['cbam']: results['cbam'][k].append(metrics_cbam[k])
        
        print(f"Top-1 Acc -> Standard: {metrics_resnet['top1']:.4f} | SE: {metrics_se['top1']:.4f} | CBAM: {metrics_cbam['top1']:.4f}\n")

    # Compile Statistics
    stats_top1 = run_statistics(results['resnet']['top1'], results['se']['top1'], results['cbam']['top1'], "Top-1 Accuracy")
    stats_f1 = run_statistics(results['resnet']['f1'], results['se']['f1'], results['cbam']['f1'], "Macro F1-Score")

    # Save to file
    with open(args.results_file, 'w') as f:
        f.write(f"=== FINAL RESULTS: {args.dataset.upper()} ===\n")
        f.write(f"Config: {args.folds}-Fold, {args.repeats} Repeats, {args.epochs} Epochs\n\n")
        
        for i in range(args.folds * args.repeats):
            f.write(f"Iteration {i+1}:\n")
            f.write(f"  Standard ResNet50: Top-1: {results['resnet']['top1'][i]:.4f}, Top-5: {results['resnet']['top5'][i]:.4f}, F1: {results['resnet']['f1'][i]:.4f}\n")
            f.write(f"  SE ResNet50:       Top-1: {results['se']['top1'][i]:.4f}, Top-5: {results['se']['top5'][i]:.4f}, F1: {results['se']['f1'][i]:.4f}\n")
            f.write(f"  CBAM ResNet50:     Top-1: {results['cbam']['top1'][i]:.4f}, Top-5: {results['cbam']['top5'][i]:.4f}, F1: {results['cbam']['f1'][i]:.4f}\n")
            f.write("-" * 50 + "\n")
            
        f.write("\nAVERAGE SCORES:\n")
        for model_name in ['resnet', 'se', 'cbam']:
            f.write(f"  {model_name.upper()} -> Top-1: {np.mean(results[model_name]['top1']):.4f} | Top-5: {np.mean(results[model_name]['top5']):.4f} | Macro-F1: {np.mean(results[model_name]['f1']):.4f}\n")

        f.write("\n" + "="*50 + "\n")
        f.write(stats_top1)
        f.write("\n" + "="*50 + "\n")
        f.write(stats_f1)

    print(stats_top1)
    print(stats_f1)
    print(f"\nExperiment complete. Results and statistical analysis saved to {args.results_file}")

if __name__ == '__main__':
    main()