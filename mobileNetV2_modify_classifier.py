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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime

# [保持原有的 SkinDataset 和 CLAHE 類別不變]
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
            
        label_dict = {'acne': 0, 'dry': 1, 'normal': 2, 'oily': 3}
        label = label_dict[label]
        
        return image, label

class CLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, img):
        img = np.array(img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)

# [保持原有的訓練和評估函數不變]
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

# 修改模型創建函數
def create_model(num_classes=4, hidden_size=512):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(hidden_size, num_classes)
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

# 創建資料集
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

# 資料分割
train_df, remaining_df = train_test_split(
    full_df,
    test_size=0.3,
    stratify=full_df['skin_type'],
    random_state=42
)

val_df, test_df = train_test_split(
    remaining_df,
    test_size=0.5,
    stratify=remaining_df['skin_type'],
    random_state=42
)

# 資料轉換
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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 創建資料載入器
train_dataset = SkinDataset(train_df, transform=transform_train)
val_dataset = SkinDataset(val_df, transform=transform)
test_dataset = SkinDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 設定實驗參數
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 30
lr = 0.001
hidden_sizes = [1024, 512, 256, 128, 64, 32]

# 創建結果資料夾
results_folder = f'model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(results_folder, exist_ok=True)

# 儲存實驗結果
results = {}

# 開始測試不同的隱藏層大小
for hidden_size in hidden_sizes:
    print(f"\n{'='*50}")
    print(f"Testing model with hidden size: {hidden_size}")
    print(f"{'='*50}")
    
    # 創建模型和優化器
    model = create_model(hidden_size=hidden_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.3,
        patience=2,
        min_lr=1e-6,
    )

    # 訓練記錄
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)
    best_val_acc = 0
    
    # 訓練迴圈
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(results_folder, f'best_model_hidden{hidden_size}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'hidden_size': hidden_size
            }, best_model_path)
        
        early_stopping(val_loss)
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
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # 最終評估
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # 儲存結果
    results[hidden_size] = {
        'final_val_acc': val_acc,
        'final_test_acc': test_acc,
        'best_val_acc': best_val_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'epochs_trained': len(train_losses)
    }
    
    # 生成混淆矩陣
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
    
    # 繪製並保存混淆矩陣
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['acne', 'dry', 'normal', 'oily'],
                yticklabels=['acne', 'dry', 'normal', 'oily'])
    plt.title(f'Confusion Matrix - Hidden Size {hidden_size}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(results_folder, f'confusion_matrix_hidden{hidden_size}.png'))
    plt.close()

# 繪製比較圖
plt.figure(figsize=(15, 10))

# 驗證準確率比較
plt.subplot(2, 1, 1)
for hidden_size in results:
    plt.plot(range(1, len(results[hidden_size]['val_accs']) + 1), 
             results[hidden_size]['val_accs'], 
             label=f'Hidden Size {hidden_size}')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.title('Validation Accuracy Comparison')
plt.legend()
plt.grid(True)

# 驗證損失比較
plt.subplot(2, 1, 2)
for hidden_size in results:
    plt.plot(range(1, len(results[hidden_size]['val_losses']) + 1), 
             results[hidden_size]['val_losses'], 
             label=f'Hidden Size {hidden_size}')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'model_comparison.png'))
plt.close()

# 保存結果報告
with open(os.path.join(results_folder, 'results_summary.txt'), 'w') as f:
    f.write("Model Architecture Comparison Results\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*60 + "\n\n")
    
    # 表格標題
    f.write(f"{'Hidden Size':<12} {'Best Val Acc':<12} {'Final Val Acc':<12} {'Test Acc':<12} {'Epochs':<8}\n")
    f.write("-"*60 + "\n")
    
    # 每個模型的結果
    for hidden_size in results:
        f.write(f"\nHidden Size: {hidden_size}\n")
        f.write(f"- Best validation accuracy: {results[hidden_size]['best_val_acc']:.2f}%\n")
        f.write(f"- Final validation accuracy: {results[hidden_size]['final_val_acc']:.2f}%\n")
        f.write(f"- Final test accuracy: {results[hidden_size]['final_test_acc']:.2f}%\n")
        f.write(f"- Number of epochs trained: {results[hidden_size]['epochs_trained']}\n")

# 打印訓練完成信息
print(f"\nTraining complete! All results have been saved to: {results_folder}")
print("\nModel checkpoints saved:")
for hidden_size in hidden_sizes:
    print(f"- best_model_hidden{hidden_size}.pth")

print("\nGenerated visualizations:")
print("- model_comparison.png")
for hidden_size in hidden_sizes:
    print(f"- confusion_matrix_hidden{hidden_size}.png")

# 打印最終比較結果
print("\nFinal Results Comparison:")
print("-" * 80)
print(f"{'Hidden Size':<12} {'Best Val Acc':<15} {'Final Val Acc':<15} {'Test Acc':<15} {'Epochs':<8}")
print("-" * 80)

# 按照驗證準確率排序結果
sorted_results = sorted(results.items(), 
                       key=lambda x: x[1]['best_val_acc'], 
                       reverse=True)

for hidden_size, result in sorted_results:
    print(f"{hidden_size:<12} "
          f"{result['best_val_acc']:.2f}%{'':<10} "
          f"{result['final_val_acc']:.2f}%{'':<10} "
          f"{result['final_test_acc']:.2f}%{'':<10} "
          f"{result['epochs_trained']:<8}")

print("-" * 80)
print("\nBest performing model:")
best_size = sorted_results[0][0]
print(f"Hidden Size: {best_size}")
print(f"Best Validation Accuracy: {results[best_size]['best_val_acc']:.2f}%")
print(f"Final Test Accuracy: {results[best_size]['final_test_acc']:.2f}%")