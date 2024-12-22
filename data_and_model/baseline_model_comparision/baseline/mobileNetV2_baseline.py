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

# 創建包含所有圖片路徑的 DataFrame
data_list = []
skin_types = ['acne', 'dry', 'normal', 'oily']
for skin_type in skin_types:
    folder_path = f'new_skin_dataset/{skin_type}'
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

# 1. 自定義資料集
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

class CLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, img):
        # 將 PIL Image 轉換為 OpenCV 格式
        img = np.array(img)
        
        # 轉換到 LAB 色彩空間
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 應用 CLAHE 到 L 通道
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        l = clahe.apply(l)
        
        # 合併通道
        lab = cv2.merge((l,a,b))
        
        # 轉回 RGB
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 轉回 PIL Image
        return Image.fromarray(img)

# 2. 資料預處理
# 增加資料增強來擴充訓練樣本
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    CLAHE(clip_limit=2.0, tile_grid_size=(8,8)),  # 添加 CLAHE
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    # ... 其他轉換
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# 3. 創建資料載入器
train_dataset = SkinDataset(train_df, transform=transform_train)
val_dataset = SkinDataset(val_df, transform=transform)
test_dataset = SkinDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 5. 將模型移至 GPU（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
used_memory = torch.cuda.memory_allocated(0)

# 取得可配置的顯存上限（單位：Byte）
max_memory = torch.cuda.get_device_properties(0).total_memory

print(f"已使用顯存: {used_memory / (1024**2):.2f} MB")
print(f"可用顯存: {max_memory / (1024**2):.2f} MB")


# 7. 訓練函數
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

# 8. 評估函數
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
    
    # 修改最後的分類層
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

# 儲存每種架構的訓練結果
results = {}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 設定訓練參數
num_epochs = 10
lr = 0.0001  # 使用之前找到的最佳學習率

# 創建新的模型實例
model = create_model().to(device)
criterion = nn.CrossEntropyLoss()
# 優化器和學習率調整
optimizer = optim.AdamW(model.parameters(), lr=lr)  # 使用AdamW和略微增加weight decay

#學習率調度器調整
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, 
#     mode='min', 
#     factor=0.2,  # 更溫和的衰減
#     patience=3,  # 減少patience
#     min_lr=1e-6,
#     verbose=True
# )

# 儲存訓練過程
train_losses = []
train_accs = []
val_losses = []
val_accs = []

early_stopping = EarlyStopping(patience=2, min_delta=1e-4)  # 增加patience和添加最小改善閾值
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

    # 根據驗證損失調整學習率
    # scheduler.step(val_loss)
    # 在調整學習率前先獲取當前學習率
    current_lr = optimizer.param_groups[0]['lr']

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    print(f'Learning Rate: {current_lr:.6f}')

# 設置圖表大小並繪製 Loss 和 Accuracy 圖
plt.figure(figsize=(12, 5))

# 繪製 Loss 圖
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(train_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# 繪製 Accuracy 圖
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_losses) + 1), train_accs, label='Train Accuracy')
plt.plot(range(1, len(train_losses) + 1), val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

# 調整子圖之間的間距
plt.tight_layout()

# 儲存 Loss 和 Accuracy 圖
plt.savefig('training_curves_mob.png')

# 顯示圖表
plt.show()
plt.close()

# 在訓練完成後，分別對驗證集和測試集進行評估
val_loss, val_acc = evaluate(model, val_loader, criterion, device)
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print('\nFinal Performance:')
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

# 在進行測試集評估後，儲存模型
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

# 為了更詳細的分析，我們可以加入混淆矩陣和分類報告
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 收集預測結果和真實標籤
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

# 繪製混淆矩陣
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['acne', 'dry', 'normal', 'oily'],
            yticklabels=['acne', 'dry', 'normal', 'oily'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# 儲存混淆矩陣圖
plt.savefig('confusion_matrix_mob.png')

plt.show()
plt.close()

# 打印分類報告
print('\nClassification Report:')
print(classification_report(y_true, y_pred, 
                          target_names=['acne', 'dry', 'normal', 'oily']))