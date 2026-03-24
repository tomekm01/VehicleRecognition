import os
import scipy.io
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


folder_path = r"C:\Users\pc\Desktop\mgr\data\StanfordCars\cars_train\cars_train"

print("Loading annotations and metadata...")
meta = scipy.io.loadmat('cars_meta.mat')
train_annos = scipy.io.loadmat('cars_train_annos.mat')

class_names = [c[0] for c in meta['class_names'][0]]
annotations = train_annos['annotations']

data = []
for anno in annotations[0]:
    data.append({
        'fname': anno['fname'][0],
        'bbox': (anno['bbox_x1'][0][0], anno['bbox_y1'][0][0], 
                 anno['bbox_x2'][0][0], anno['bbox_y2'][0][0]),
        'class_id': anno['class'][0][0],
        'class_name': class_names[anno['class'][0][0] - 1]
    })

df = pd.DataFrame(data)

df_subset = df[df['class_id'].isin([1, 2, 3])]

df_experiment = df_subset.sample(n=100, random_state=42).reset_index(drop=True)

class_mapping = {1: 0, 2: 1, 3: 2}
df_experiment['mapped_label'] = df_experiment['class_id'].map(class_mapping)

df_train = df_experiment.iloc[:99].reset_index(drop=True)
df_test = df_experiment.iloc[99:].reset_index(drop=True)

test_img_name = df_test.iloc[0]['fname']
test_class_name = df_test.iloc[0]['class_name']
print(f"Chosen 99 training photos.")
print(f"Test photo: {test_img_name} ({test_class_name})\n")


# 2. DATA DEFINITION

class StanfordCarsDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        clean_fname = str(row['fname'])[-9:]
        
        img_path = Path(self.image_dir) / clean_fname
        
        image = Image.open(img_path).convert("RGB")
        image = image.crop(row['bbox'])
        label = row['mapped_label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),       
    transforms.ToTensor(),               
    transforms.Normalize(                
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = StanfordCarsDataset(df_train, image_dir=folder_path, transform=transform)
test_dataset = StanfordCarsDataset(df_test, image_dir=folder_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Change last layer

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3) 

model = model.to(device)

# Loss function + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5 
print("\nTraining...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Acc: {epoch_acc:.2f}%")

print("\nLeave-One-Out...")
model.eval()

reverse_mapping = {v: k for k, v in class_mapping.items()}

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        pred_label_id = predicted.item()
        true_label_id = labels.item()
        
        original_pred_id = reverse_mapping[pred_label_id]
        original_true_id = reverse_mapping[true_label_id]
        
        pred_class_name = class_names[original_pred_id - 1]
        
        print(f"--> Guess: {pred_class_name}")
        print(f"--> Real: {test_class_name}")
        
        if original_pred_id == original_true_id:
            print("SUCCESS!")
        else:
            print("FAILURE.")