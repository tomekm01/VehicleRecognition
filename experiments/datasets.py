from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Dataset & transforms

class CarsDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(self.img_dir) / row['filename']
        image = Image.open(img_path).convert("RGB")
        
        if all(col in self.df.columns for col in ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']):
            bbox = (row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2'])
            image = image.crop(bbox)

        if self.transform:
            image = self.transform(image)
            
        label = row['mapped_label'] 
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