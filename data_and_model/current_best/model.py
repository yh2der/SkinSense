import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class SkinDataset(Dataset):
    """皮膚圖像數據集類"""
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.label_dict = {'acne': 0, 'dry': 1, 'normal': 2, 'oily': 3}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['skin_type']
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = self.label_dict[label]
        return image, label

class SkinClassifier(nn.Module):
    """皮膚分類模型類"""
    def __init__(self, num_classes=4):
        super(SkinClassifier, self).__init__()
        self.model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.last_channel, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class EarlyStopping:
    """早停機制類"""
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

class SkinClassifierTrainer:
    """模型訓練器類"""
    def __init__(self, model, device, criterion=None, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return running_loss/len(loader), 100.*correct/total
    
    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return running_loss/len(loader), 100.*correct/total
    
    def train(self, train_loader, val_loader, num_epochs, early_stopping=None):
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            if early_stopping:
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            if self.scheduler:
                print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        return train_losses, train_accs, val_losses, val_accs

class DataPreparation:
    """數據準備類"""
    @staticmethod
    def get_transforms():
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return transform_train, transform_val
    
    @staticmethod
    def prepare_data(data_dir):
        data_list = []
        skin_types = ['acne', 'dry', 'normal', 'oily']
        
        for skin_type in skin_types:
            folder_path = os.path.join(data_dir, skin_type)
            image_files = os.listdir(folder_path)
            
            for img_file in image_files:
                data_list.append({
                    'skin_type': skin_type,
                    'image_path': os.path.join(folder_path, img_file)
                })
        
        full_df = pd.DataFrame(data_list)
        
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
        
        return train_df, val_df, test_df

class Visualization:
    """視覺化工具類"""
    @staticmethod
    def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path=None):
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
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path=None):
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['acne', 'dry', 'normal', 'oily'],
                   yticklabels=['acne', 'dry', 'normal', 'oily'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

def main():
    # 設置設備
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 準備數據
    data_prep = DataPreparation()
    transform_train, transform_val = data_prep.get_transforms()
    train_df, val_df, test_df = data_prep.prepare_data('data_and_model/data/dataset')
    
    # 創建數據集和數據加載器
    train_dataset = SkinDataset(train_df, transform=transform_train)
    val_dataset = SkinDataset(val_df, transform=transform_val)
    test_dataset = SkinDataset(test_df, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 創建模型和訓練器
    model = SkinClassifier().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=2, min_lr=1e-7
    )
    trainer = SkinClassifierTrainer(model, device, optimizer=optimizer, scheduler=scheduler)
    
    # 訓練模型
    early_stopping = EarlyStopping(patience=5, min_delta=1e-3)
    train_losses, train_accs, val_losses, val_accs = trainer.train(
        train_loader, val_loader, num_epochs=50, early_stopping=early_stopping
    )
    
    # 評估最終性能
    val_loss, val_acc = trainer.evaluate(val_loader)
    test_loss, test_acc = trainer.evaluate(test_loader)
    
    print('\nFinal Performance:')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    # 保存模型
    model_save_path = f'skin_classifier_acc_{test_acc:.2f}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'val_acc': val_acc,
        'epoch': len(train_losses),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }, model_save_path)
    
    # 繪製訓練曲線
    Visualization.plot_training_curves(
        train_losses, train_accs, val_losses, val_accs,
        save_path='training_curves_mob.png'
    )
    
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
    
    Visualization.plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix_mob.png')
    
    # 生成分類報告
    with open('classification_report.txt', 'w', encoding='utf-8') as f:
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Classification Report - {current_time}\n")
        f.write('-' * 60 + '\n')
        report = classification_report(
            y_true, y_pred,
            target_names=['acne', 'dry', 'normal', 'oily'],
            digits=3
        )
        f.write(report)
        f.write('-' * 60 + '\n')

if __name__ == '__main__':
    main()