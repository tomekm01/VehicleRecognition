

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np

# Configuration
CSV_PATH = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\StanfordCars_devkit\cars_train.csv"
IMG_DIR = r"C:\Users\pc\Desktop\pracamag\VehicleRecognition\data\StanfordCars\cars_train\cars_train"
NUM_CLASSES = 196
BATCH_SIZE = 32
EPOCHS = 5
FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = 'experiment_results.txt'

# CBAM IMPLEMENTATION (from official github repo)
class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = torch.nn.functional.adaptive_max_pool2d(x, output_size=1)
        avg = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = torch.sigmoid(output) * x 
        return output

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = torch.sigmoid(output) * x 
        return output 

class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x


# DATASET & MODELS
class CarsDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(self.img_dir) / row['filename']
        
        image = Image.open(img_path).convert("RGB")
        bbox = (row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2'])
        image = image.crop(bbox)

        if self.transform:
            image = self.transform(image)
            
        label = row['class_id'] - 1 
        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_standard_resnet():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model

def create_cbam_resnet():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    model.layer1 = nn.Sequential(model.layer1, CBAM(channels=256, r=16))
    model.layer2 = nn.Sequential(model.layer2, CBAM(channels=512, r=16))
    model.layer3 = nn.Sequential(model.layer3, CBAM(channels=1024, r=16))
    model.layer4 = nn.Sequential(model.layer4, CBAM(channels=2048, r=16))
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model


# TRAINING LOOP
def train_and_evaluate(model, train_loader, val_loader):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
    accuracy = correct / total
    
    del model
    torch.cuda.empty_cache()
    
    return accuracy

def main():
    print(f"Device: {DEVICE}")
    print(f"Starting {FOLDS}-Fold Cross Validation (Epochs per fold: {EPOCHS})\n")
    
    full_dataset = CarsDataset(CSV_PATH, IMG_DIR)
    kfold = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    results_resnet = []
    results_cbam = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(full_dataset)))):
        print(f"--- Fold {fold + 1}/{FOLDS} ---")
        
        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        
        train_sub.dataset.transform = train_transform
        val_sub.dataset.transform = val_transform
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print("Training Standard ResNet50...")
        model_std = create_standard_resnet()
        acc_resnet = train_and_evaluate(model_std, train_loader, val_loader)
        results_resnet.append(acc_resnet)
        print(f"ResNet50 Fold {fold + 1} Acc: {acc_resnet:.4f}")
        
        print("Training Official GitHub CBAM ResNet50...")
        model_cbam = create_cbam_resnet()
        acc_cbam = train_and_evaluate(model_cbam, train_loader, val_loader)
        results_cbam.append(acc_cbam)
        print(f"CBAM ResNet50 Fold {fold + 1} Acc: {acc_cbam:.4f}\n")

    avg_resnet = np.mean(results_resnet)
    avg_cbam = np.mean(results_cbam)
    
    # Print results to console
    print("=========================================")
    print("FINAL CROSS-VALIDATION RESULTS")
    print("=========================================")
    print(f"Standard ResNet50 Avg Accuracy: {avg_resnet:.4f}")
    print(f"GitHub CBAM ResNet50 Avg Accuracy: {avg_cbam:.4f}")
    
    # Save results to text file
    with open(RESULTS_FILE, 'w') as f:
        f.write("=========================================\n")
        f.write("FINAL CROSS-VALIDATION RESULTS\n")
        f.write("=========================================\n")
        f.write(f"Configuration: {FOLDS}-Fold CV, {EPOCHS} Epochs per fold\n\n")
        
        for i in range(FOLDS):
            f.write(f"Fold {i+1}:\n")
            f.write(f"  Standard ResNet50: {results_resnet[i]:.4f}\n")
            f.write(f"  CBAM ResNet50:     {results_cbam[i]:.4f}\n")
            f.write("-" * 40 + "\n")
            
        f.write("\nAVERAGE ACCURACY:\n")
        f.write(f"  Standard ResNet50: {avg_resnet:.4f}\n")
        f.write(f"  CBAM ResNet50:     {avg_cbam:.4f}\n")

    print(f"\nResults have been successfully saved to {RESULTS_FILE}")

if __name__ == '__main__':
    main()