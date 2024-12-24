# SkinSense

一個讓使用者上傳照片分析自身是甚麼膚質的圖像分類模型，並且給予適合的保養品成分跟日常習慣建議

## 功能

- 提供使用者上傳照片進行分析
    - 痘痘肌
    - 油肌
    - 正常肌
    - 乾肌
    
- 給予能適用的保養品成份、日常保養建議以及推薦適合使用者的商品


## 技術

- 後端框架：Flask
- 深度學習框架：PyTorch
- 圖像處理：OpenCV
- 前端技術：
    - HTML
    - CSS3
    - JavaScript

## 安裝指南

在您的虛擬環境中使用 python 3.10，搭配以下指令下載所需套件
```python=
pip install -r requirements.txt
```

## 使用方法

1. 啟動服務器
    ```python=
    python app.py
    ```

2. 開啟瀏覽器訪問：http://127.0.0.1:5000

3. 使用步驟：
    - 點擊「上傳圖片」按鈕選擇要分析的臉部照片 (首頁能瀏覽所有商品)
    - 系統會自動進行膚質分析
    - 顯示分析結果與建議的保養成分
    - 可查看推薦商品
    - 可選擇下載詳細的 PDF 報告

## 資料集

使用 [face-skin-type](https://www.kaggle.com/datasets/muttaqin1113/face-skin-type) 和 [oily-dry-and-normal-skin-types-dataset](https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset) 混合，先對 先對 oily-dry-and-normal-skin-types-dataset 的 normal 類別做臉部提取，再將提取的結果分成臉部各個區域，最後進行資料清理

## 模型資訊
### baseline

利用預訓練模型 ResNet18、efficientNetB0、mobileNetV2，並且先比較這三種 baseline 效能，選擇方式是因為資料量不大，加上未來有機會佈署在行動裝置上，因此選擇參數量較小的模型來測試

#### [結果報告](/data_and_model/current_best/classification_report.txt)

![alt](/data_and_model/baseline_model_comparision/analysis/training_curves_efficientnet.png)
![alt](/data_and_model/baseline_model_comparision/analysis/training_curves_mobilenet.png)
![alt](/data_and_model/baseline_model_comparision/analysis/training_curves_resnet18.png)

![alt](/data_and_model/baseline_model_comparision/analysis/confusion_matrix_efficientnet.png)
![alt](/data_and_model/baseline_model_comparision/analysis/confusion_matrix_mobilenet.png)
![alt](/data_and_model/baseline_model_comparision/analysis/confusion_matrix_resnet18.png)

### chosen model
#### [結果報告](/data_and_model/current_best/classification_report.txt)
![alt](/data_and_model/current_best/training_curves_mob.png)
![alt](/data_and_model/current_best/confusion_matrix_mob.png)

## Demo

### 首頁 
![alt text](image.png)
#### 首頁可查看所有商品
![alt text](/demo_pics/image-5.png)
### 選擇照片後
![alt text](/demo_pics/image-1.png)
### 按下分析後
![alt text](/demo_pics/image-2.png)
### 下載報告
![alt text](/demo_pics/image-4.png)
### 推薦適合產品
![alt text](/demo_pics/image-3.png)
