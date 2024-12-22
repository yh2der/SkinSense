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
from torchvision.models import (
    ResNet18_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights
)
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 創建包含所有圖片路徑的 DataFrame
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

# 轉換成 DataFrame
full_df = pd.DataFrame(data_list)

# First split: separate training set (70%) and remaining data (30%)
train_df, remaining_df = train_test_split(
    full_df,
    test_size=0.3,
    stratify=full_df['skin_type'],
    random_state=42
)

# Second split: split remaining data into validation (15%) and test sets (15%)
val_df, test_df = train_test_split(
    remaining_df,
    test_size=0.5,
    stratify=remaining_df['skin_type'],
    random_state=42
)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0

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

# 資料預處理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    # CLAHE(clip_limit=2.0, tile_grid_size=(8,8)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.ColorJitter(
    #     brightness=0.2,
    #     contrast=0.2,
    #     saturation=0.2,
    #     hue=0.1
    # ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 模型創建函數
def create_resnet18():
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)     
    num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(
    #     nn.Dropout(0.3),
    #     nn.Linear(num_ftrs, 4)
    # )
    model.fc = nn.Linear(num_ftrs, 4)

    return model, 'resnet18'

def create_efficientnet():
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    # model.classifier[1] = nn.Sequential(
    #     nn.Dropout(0.3),
    #     nn.Linear(num_ftrs, 4)
    # )
    model.classifier[1] = nn.Linear(num_ftrs, 4)
    return model, 'efficientnet'

def create_mobilenet():
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    # model.classifier[1] = nn.Sequential(
    #     nn.Dropout(0.3),
    #     nn.Linear(num_ftrs, 4)
    # )
    model.classifier[1] = nn.Linear(num_ftrs, 4)
    return model, 'mobilenet'

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

def train_model(model, model_name, train_loader, val_loader, test_loader, 
                num_epochs=10, lr=0.00005, device='cuda'):
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.1, patience=2, verbose=True
    # )
    
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    early_stopping = EarlyStopping(patience=3, min_delta=1e-4)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # scheduler.step(val_loss)
        early_stopping(val_loss, model)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'{model_name} - Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        if early_stopping.early_stop:
            print(f"{model_name}: Early stopping triggered")
            model.load_state_dict(early_stopping.best_model)
            break
    
    # 繪製訓練曲線
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} - Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'analysis/training_curves_{model_name}.png')
    plt.close()
    
    # 評估最終性能
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f'\n{model_name} Final Performance:')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    # 儲存模型
    model_save_path = f'models/skin_classifier_{model_name}_acc_{test_acc:.2f}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'val_acc': val_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }, model_save_path)
    
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
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['acne', 'dry', 'normal', 'oily'],
                yticklabels=['acne', 'dry', 'normal', 'oily'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'analysis/confusion_matrix_{model_name}.png')
    plt.close()
    
    # 儲存分類報告
    report = classification_report(y_true, y_pred, 
                                 target_names=['acne', 'dry', 'normal', 'oily'])
    with open(f'analysis/classification_report_{model_name}.txt', 'w') as f:
        f.write(report)
    
    return test_acc, model_save_path

# 主程式
if __name__ == "__main__":
    # 確保存放結果的資料夾存在
    os.makedirs('analysis', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 設定裝置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 設定基本參數
    num_epochs = 30
    base_lr = 0.00005
    
    # 批次大小設定
    batch_sizes = {
        'resnet18': 32,  # 可以試試64
        'efficientnet': 64,  # 增加batch size提高穩定性
        'mobilenet': 16   # 降低batch size改善泛化性能
    }
    
    # 建立資料集
    train_dataset = SkinDataset(train_df, transform=transform_train)
    val_dataset = SkinDataset(val_df, transform=transform_val)
    test_dataset = SkinDataset(test_df, transform=transform_val)
    
    # 訓練所有模型
    model_creators = [create_resnet18, create_efficientnet, create_mobilenet]
    model_results = {}
    
    for create_fn in model_creators:
        model, model_name = create_fn()
        print(f"\nTraining {model_name}...")
        
        batch_size = batch_sizes[model_name]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        test_acc, model_path = train_model(
            model, model_name, 
            train_loader, val_loader, test_loader,
            num_epochs=num_epochs,
            lr=base_lr,
            device=device
        )
        
        model_results[model_name] = {
            'accuracy': test_acc,
            'model_path': model_path
        }

    # 打印最終比較結果
    print("\nFinal Comparison of Models:")
    for model_name, results in model_results.items():
        print(f"{model_name}: Test Accuracy = {results['accuracy']:.2f}%")