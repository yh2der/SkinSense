from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sqlite3
from data_and_model.models.model import SkinClassifier
import logging
import traceback
from flask_cors import CORS

# 初始化日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def get_db_connection():
    conn = sqlite3.connect('database/skincare.db')
    conn.row_factory = sqlite3.Row
    return conn

def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SkinClassifier().to(device)
    
    checkpoint = torch.load('data_and_model/current_best/skin_classifier_acc_94.86.pth',
                          map_location=device, weights_only=True)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def get_recommendations(skin_type=None, min_price=None, max_price=None, sort_by=None):
    conn = sqlite3.connect('database/skincare.db')
    cursor = conn.cursor()

    query = "SELECT name, price FROM products"
    params = []
    where_clauses = []

    if skin_type:
        where_clauses.append("type = ?")
        params.append(skin_type)

    if min_price is not None:
        where_clauses.append("price >= ?")
        params.append(min_price)

    if max_price is not None:
        where_clauses.append("price <= ?")
        params.append(max_price)

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    if sort_by == 'price-asc':
        query += " ORDER BY price ASC"
    elif sort_by == 'price-desc':
        query += " ORDER BY price DESC"

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()

    return [{'name': row[0], 'price': row[1]} for row in results]

# 設置設備和模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/products')
def products():
    skin_type = request.args.get('type')
    return render_template('products.html', skin_type=skin_type)

@app.route('/api/products')
def get_products():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM products')
    products = cursor.fetchall()
    
    products_list = []
    for product in products:
        products_list.append({
            'id': product['id'],
            'name': product['name'],
            'price': product['price'],
            'type': product['type']
        })
    
    conn.close()
    return jsonify(products_list)

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received prediction request")
    
    if 'image' not in request.files:
        logger.error("No image in request")
        return jsonify({'error': 'No image uploaded'}), 400
    
    try:
        image_file = request.files['image']
        logger.info(f"Processing image: {image_file.filename}")
        
        if image_file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
        
        if not model:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not initialized'}), 500
            
        img = Image.open(image_file).convert('RGB')
        logger.info(f"Image opened, size: {img.size}")
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        logger.info(f"Image transformed to tensor: {img_tensor.shape}")
        
        with torch.no_grad():
            outputs = model(img_tensor)
            logger.info(f"Model output shape: {outputs.shape}")
            probabilities = outputs.softmax(1)
            confidence, predicted = torch.max(probabilities, 1)
            logger.info(f"Prediction: {predicted.item()}, Confidence: {confidence.item()}")
        
        result = predicted.item()
        confidence_score = confidence.item()
        
        class_names = ['acne', 'dry', 'normal', 'oily']
        prediction = class_names[result]
        chinese_names = {
            'acne': '痘痘肌',
            'dry': '乾性肌膚',
            'normal': '中性肌膚',
            'oily': '油性肌膚'
        }
        chinese_prediction = chinese_names[prediction]
        
        recommendations = get_recommendations(prediction)
        logger.info(f"Got {len(recommendations)} recommendations")
        
        return jsonify({
            'prediction': prediction,
            'chinese': chinese_prediction,
            'confidence': float(confidence_score),
            'recommendations': recommendations
        })
        
    except Exception as e:
        error_msg = traceback.format_exc()
        logger.error(f"Error in prediction: {error_msg}")
        return jsonify({
            'error': str(e),
            'traceback': error_msg
        }), 500

if __name__ == '__main__':
    app.run(debug=True)