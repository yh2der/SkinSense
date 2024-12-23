import os
from pathlib import Path
import cv2
import numpy as np
from retinaface import RetinaFace

class FaceProcessor:
    def __init__(self, base_path="dataset", output_path="processed_dataset"):
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.categories = ['dry', 'normal', 'oily']
        
    def detect_face(self, image):
        """使用 RetinaFace 檢測人臉和關鍵點"""
        try:
            faces = RetinaFace.detect_faces(image)
            if isinstance(faces, dict):
                # 獲取最大的臉部區域
                max_area = 0
                max_face = None
                max_landmarks = None
                
                for key in faces:
                    face = faces[key]
                    bbox = face["facial_area"]
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_area = area
                        max_face = bbox
                        max_landmarks = face["landmarks"]
                
                if max_face is not None and max_landmarks is not None:
                    return max_face, max_landmarks
                    
            return None, None
                
        except Exception as e:
            print(f"臉部檢測錯誤: {e}")
            return None, None

    def crop_face_regions(self, image, bbox, landmarks):
        """裁剪臉部不同區域"""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        
        regions = {}
        
        # 計算各區域的裁剪範圍
        # 額頭區域 (使用眼睛位置的上方)
        forehead_height = int(height * 0.3)
        regions['forehead'] = (
            x1,
            max(0, y1),
            x2,
            y1 + forehead_height
        )
        
        # 鼻子區域 (使用鼻子關鍵點為中心)
        nose_x = landmarks['nose'][0]
        nose_y = landmarks['nose'][1]
        nose_size = int(width * 0.3)
        regions['nose'] = (
            int(nose_x - nose_size/2),
            int(nose_y - nose_size/2),
            int(nose_x + nose_size/2),
            int(nose_y + nose_size/2)
        )
        
        # 左右臉頰
        cheek_width = int(width * 0.3)
        cheek_height = int(height * 0.4)
        regions['cheek_left'] = (
            x1,
            y1 + int(height * 0.3),
            x1 + cheek_width,
            y1 + int(height * 0.7)
        )
        regions['cheek_right'] = (
            x2 - cheek_width,
            y1 + int(height * 0.3),
            x2,
            y1 + int(height * 0.7)
        )
        
        # 下巴區域
        chin_height = int(height * 0.3)
        regions['chin'] = (
            x1,
            y2 - chin_height,
            x2,
            y2
        )
        
        # 裁剪並調整大小
        crops = {}
        for region_name, (rx1, ry1, rx2, ry2) in regions.items():
            # 確保座標在圖片範圍內
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(image.shape[1], rx2), min(image.shape[0], ry2)
            
            if rx2 > rx1 and ry2 > ry1:  # 確保有有效的裁剪區域
                crop = image[ry1:ry2, rx1:rx2]
                # 調整為正方形
                crops[region_name] = cv2.resize(crop, (224, 224))
        
        return crops

    def process_image(self, image_path, output_dir):
        """處理單張圖片"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise Exception("無法讀取圖片")
            
            # 檢測臉部和關鍵點
            bbox, landmarks = self.detect_face(image)
            if bbox is None or landmarks is None:
                raise Exception("未檢測到臉部")
            
            # 裁剪各個區域
            crops = self.crop_face_regions(image, bbox, landmarks)
            
            # 儲存裁剪後的圖片
            filename = image_path.stem
            for region_name, crop in crops.items():
                output_path = output_dir / f"{filename}_{region_name}.jpg"
                cv2.imwrite(str(output_path), crop)
            
            return True
            
        except Exception as e:
            print(f"處理圖片時發生錯誤 {image_path}: {e}")
            return False

    def process_dataset(self):
        """處理整個數據集"""
        # 創建輸出目錄
        for split in ['train', 'valid', 'test']:
            for category in self.categories:
                output_dir = self.output_path / split / category
                output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {'success': 0, 'failed': 0, 'total': 0}
        
        # 處理每個分割的數據
        for split in ['train', 'valid', 'test']:
            print(f"\n處理 {split} 集...")
            for category in self.categories:
                input_dir = self.base_path / split / category
                output_dir = self.output_path / split / category
                
                if input_dir.exists():
                    for img_path in input_dir.glob('*'):
                        stats['total'] += 1
                        if self.process_image(img_path, output_dir):
                            stats['success'] += 1
                        else:
                            stats['failed'] += 1
        
        # 輸出處理統計
        print("\n處理完成統計:")
        print(f"總圖片數: {stats['total']}")
        print(f"成功處理: {stats['success']}")
        print(f"處理失敗: {stats['failed']}")
        print(f"成功率: {(stats['success']/stats['total']*100):.2f}%")
        
        return stats