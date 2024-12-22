import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['skin_type']
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # 將標籤轉換為數字
        label_dict = {'acne': 0, 'dry': 1, 'normal': 2, 'oily': 3}
        label = label_dict[label]
        
        return image, label

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(loader), 100.*correct/total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss/len(loader), 100.*correct/total

def create_model(num_classes=4):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
 
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, num_classes)
    )
    return model

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

data_list = []
skin_types = ['acne', 'dry', 'normal', 'oily']
for skin_type in skin_types:
    folder_path = f'data/dataset/{skin_type}'
    image_files = os.listdir(folder_path)
    
    for img_file in image_files:
        data_list.append({
            'skin_type': skin_type,
            'image_path': os.path.join(folder_path, img_file)
        })

full_df = pd.DataFrame(data_list)

# First split: separate training set (70%) and remaining data (30%)
train_df, remaining_df = train_test_split(
    full_df,
    test_size=0.3,  # 30% for remaining data
    stratify=full_df['skin_type'],
    random_state=42
)

# Second split: split remaining data into validation (15%) and test sets (15%)
val_df, test_df = train_test_split(
    remaining_df,
    test_size=0.5,  # Split remaining data equally
    stratify=remaining_df['skin_type'],
    random_state=42
)

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10), 
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

train_dataset = SkinDataset(train_df, transform=transform_train)
val_dataset = SkinDataset(val_df, transform=transform)
test_dataset = SkinDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

results = {}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 50
lr = 0.00005

model = create_model().to(device)

criterion = nn.CrossEntropyLoss()


optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)  # 使用AdamW和略微增加weight decay

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',  
    factor=0.3,
    patience=2,
    min_lr=1e-7,
)

train_losses = []
train_accs = []
val_losses = []
val_accs = []

early_stopping = EarlyStopping(patience=5, min_delta=1e-3,)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    
    early_stopping(val_loss)    
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]['lr']

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f'Learning Rate: {current_lr:.6f}')


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(train_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_losses) + 1), train_accs, label='Train Accuracy')
plt.plot(range(1, len(train_losses) + 1), val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig('training_curves_mob.png')

plt.show()
plt.close()

val_loss, val_acc = evaluate(model, val_loader, criterion, device)
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print('\nFinal Performance:')
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

model_save_path = f'skin_classifier_acc_{test_acc:.2f}.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'test_acc': test_acc,
    'val_acc': val_acc,
    'epoch': epoch,
    'train_losses': train_losses,
    'train_accs': train_accs,
    'val_losses': val_losses,
    'val_accs': val_accs
}, model_save_path)

print(f'\nModel saved as: {model_save_path}')

y_pred = []
y_true = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['acne', 'dry', 'normal', 'oily'],
            yticklabels=['acne', 'dry', 'normal', 'oily'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.savefig('confusion_matrix_mob.png')

plt.show()
plt.close()

with open('classification_report.txt', 'w', encoding='utf-8') as f:
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"Classification Report - {current_time}\n")
    f.write('-' * 60 + '\n')

    report = classification_report(y_true, y_pred,
                                 target_names=['acne', 'dry', 'normal', 'oily'],
                                 digits=3)
    f.write(report)
    
    f.write('-' * 60 + '\n')

print("Classification report has been saved to 'classification_report.txt'")