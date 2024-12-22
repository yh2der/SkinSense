import os
from pathlib import Path
import cv2
import numpy as np
from collections import Counter
from retinaface import RetinaFace

class SkinTypeDataProcessor:
    def __init__(self, base_path="Oily-Dry-Skin-Types", output_path="Oily-Dry-Skin-Types-Faces"):
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.categories = ['dry', 'normal', 'oily']
        
    def detect_and_crop_face(self, image, padding=0.1):
        """使用 RetinaFace 偵測並精確裁剪臉部"""
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
                    x1, y1, x2, y2 = max_face
                    
                    # 使用臉部關鍵點調整裁剪範圍
                    left_eye = max_landmarks["left_eye"]
                    right_eye = max_landmarks["right_eye"]
                    nose = max_landmarks["nose"]
                    
                    # 計算臉部中心
                    face_center_x = (left_eye[0] + right_eye[0]) / 2
                    face_center_y = nose[1]
                    
                    # 計算眼睛之間的距離來作為縮放基準
                    eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + 
                                        (right_eye[1] - left_eye[1])**2)
                    
                    # 根據眼距計算臉部範圍
                    face_size = int(eye_distance * 2.2)  # 調整這個係數可以改變裁剪範圍
                    
                    # 計算新的裁剪範圍
                    new_x1 = int(face_center_x - face_size/2)
                    new_x2 = int(face_center_x + face_size/2)
                    new_y1 = int(face_center_y - face_size/2)
                    new_y2 = int(face_center_y + face_size/2)
                    
                    # 加入最小的padding
                    padding_size = int(face_size * padding)
                    final_x1 = max(new_x1 - padding_size, 0)
                    final_y1 = max(new_y1 - padding_size, 0)
                    final_x2 = min(new_x2 + padding_size, image.shape[1])
                    final_y2 = min(new_y2 + padding_size, image.shape[0])
                    
                    # 裁剪臉部區域
                    face = image[final_y1:final_y2, final_x1:final_x2]
                    
                    # 確保裁剪後的圖片不是空的
                    if face.size > 0:
                        # 調整為正方形，使用白色背景
                        height, width = face.shape[:2]
                        max_dim = max(height, width)
                        
                        # 創建白色正方形畫布
                        square_face = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255
                        
                        # 將臉部圖片置中
                        y_offset = (max_dim - height) // 2
                        x_offset = (max_dim - width) // 2
                        square_face[y_offset:y_offset+height, x_offset:x_offset+width] = face
                        
                        return square_face, True
                        
                return None, False
                
            return None, False
                
        except Exception as e:
            print(f"臉部檢測錯誤: {e}")
            return None, False

    def process_and_save_dataset(self, target_size=(224, 224)):
        """處理整個數據集並保存裁剪後的臉部圖片"""
        # 創建輸出目錄結構
        for split in ['train', 'valid', 'test']:
            for category in self.categories:
                output_dir = self.output_path / split / category
                output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {'success': 0, 'failed': 0, 'total': 0}
        failed_images = []
        
        # 處理每個分割的數據
        for split in ['train', 'valid', 'test']:
            print(f"\n處理 {split} 集...")
            for category in self.categories:
                input_dir = self.base_path / split / category
                output_dir = self.output_path / split / category
                
                if input_dir.exists():
                    for img_path in input_dir.glob('*'):
                        stats['total'] += 1
                        try:
                            # 讀取圖片
                            img = cv2.imread(str(img_path))
                            if img is None:
                                raise Exception("無法讀取圖片")
                            
                            # 偵測並裁剪臉部
                            face_img, face_detected = self.detect_and_crop_face(img)
                            
                            if face_detected:
                                # 調整大小
                                face_resized = cv2.resize(face_img, target_size)
                                
                                # 保存處理後的圖片
                                output_path = output_dir / img_path.name
                                cv2.imwrite(str(output_path), face_resized)
                                stats['success'] += 1
                                
                                if stats['total'] % 50 == 0:
                                    print(f"已處理: {stats['total']} 張圖片")
                                    print(f"當前成功率: {(stats['success']/stats['total']*100):.2f}%")
                            else:
                                stats['failed'] += 1
                                failed_images.append(str(img_path))
                                
                        except Exception as e:
                            stats['failed'] += 1
                            failed_images.append(str(img_path))
                            print(f"處理圖片時發生錯誤 {img_path}: {e}")
        
        # 輸出處理統計
        print("\n處理完成統計:")
        print(f"總圖片數: {stats['total']}")
        print(f"成功處理: {stats['success']}")
        print(f"處理失敗: {stats['failed']}")
        print(f"成功率: {(stats['success']/stats['total']*100):.2f}%")
        
        # 保存失敗的圖片清單
        if failed_images:
            with open('failed_images.txt', 'w') as f:
                for img_path in failed_images:
                    f.write(f"{img_path}\n")
            print("\n失敗的圖片清單已保存到 failed_images.txt")
        
        return stats
    
# 初始化處理器
processor = SkinTypeDataProcessor()

# 處理數據集
stats = processor.process_and_save_dataset()

# 驗證結果
processor.verify_processed_dataset()