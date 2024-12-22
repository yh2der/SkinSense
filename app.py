from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

from data_and_model.models.model import SkinClassifier

app = Flask(__name__)

def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SkinClassifier().to(device)  # 使用重構後的類名
    
    # 載入模型權重
    checkpoint = torch.load('data_and_model/current_best/skin_classifier_acc_94.86.pth',
                          map_location=device, weights_only=True)
    
    # 檢查 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

app = Flask(__name__)

# 設置設備
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 圖像轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 載入模型
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    try:
        # 獲取上傳的圖片
        image_file = request.files['image']
        img = Image.open(image_file).convert('RGB')  # 確保圖片是 RGB 格式
        
        # 預處理圖片
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 進行預測
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = outputs.softmax(1)
            confidence, predicted = torch.max(probabilities, 1)
            
        # 獲取預測結果
        result = predicted.item()
        confidence_score = confidence.item()
        
        class_names = ['acne', 'dry', 'normal', 'oily']
        chinese_names = {
            'acne': '痘痘肌',
            'dry': '乾性肌膚',
            'normal': '中性肌膚',
            'oily': '油性肌膚'
        }
        prediction = class_names[result]
        chinese_prediction = chinese_names[prediction]

        return jsonify({
            'prediction': prediction,
            'chinese': chinese_prediction,
            'confidence': float(confidence_score)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)