import cv2
import mediapipe as mp
import os
import numpy as np
import glob
import logging
import random

# 設置logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TF 警告

class FaceCropper:
    def __init__(self, preview_mode=False):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.preview_mode = preview_mode

    def detect_face(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        results = self.face_detection.process(image_rgb)
        
        if not results.detections:
            return None
        
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        return (x, y, width, height)

    def get_crop_region(self, face_bbox, image_width, image_height, region_type):
        """基於臉部位置獲取特定區域的裁剪範圍"""
        x, y, width, height = face_bbox
        crop_size = int(height * 0.75)
        
        # 根據不同區域類型定義裁剪位置
        if region_type == 'forehead':
            center_y = y + height//6  # 額頭位置
        elif region_type == 'nose':
            center_y = y + height//2  # 鼻子位置
        elif region_type == 'chin':
            center_y = y + height*4//5  # 下巴位置
        elif region_type == 'cheek_left':
            center_y = y + height//2
            x = x - width//4  # 左臉頰
        elif region_type == 'cheek_right':
            center_y = y + height//2
            x = x + width*5//4  # 右臉頰
        else:
            center_y = y + height//2  # 默認中心位置
        
        center_x = x + width//2
        
        # 確保裁剪區域在圖片範圍內
        x1 = max(0, center_x - crop_size//2)
        y1 = max(0, center_y - crop_size//2)
        x2 = min(image_width, center_x + crop_size//2)
        y2 = min(image_height, center_y + crop_size//2)
        
        return (x1, y1, x2, y2), region_type

    def process_image(self, image_path, output_dir):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"無法讀取圖片: {image_path}")
                return False
                
            h, w = image.shape[:2]
            
            # 檢測臉部
            face_bbox = self.detect_face(image)
            if face_bbox is None:
                print(f"未檢測到人臉: {image_path}")
                return False
            
            # 定義要處理的區域類型
            region_types = ['forehead', 'nose', 'chin', 'cheek_left', 'cheek_right']
            processed_count = 0
            
            for region_type in region_types:
                # 獲取特定區域的裁剪範圍
                region, region_name = self.get_crop_region(face_bbox, w, h, region_type)
                if region is None:
                    continue
                    
                x1, y1, x2, y2 = region
                filename = os.path.basename(image_path)
                name_without_ext = os.path.splitext(filename)[0]
                
                if self.preview_mode:
                    preview_img = image.copy()
                    cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(preview_img, region_name, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    display_height = 800
                    aspect_ratio = image.shape[1] / image.shape[0]
                    display_width = int(display_height * aspect_ratio)
                    preview_img_resized = cv2.resize(preview_img, (display_width, display_height))
                    
                    cv2.imshow('Preview', preview_img_resized)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        return False
                    cv2.destroyAllWindows()
                
                # 裁剪並保存圖片
                cropped = image[y1:y2, x1:x2]
                if cropped.size == 0:
                    print(f"裁剪區域無效: {image_path} - {region_type}")
                    continue
                
                # 調整輸出大小
                target_size = (224, 224)
                cropped = cv2.resize(cropped, target_size)
                
                # 保存裁剪後的圖片
                output_path = os.path.join(output_dir, f"{name_without_ext}_{region_type}.jpg")
                cv2.imwrite(output_path, cropped)
                processed_count += 1
            
            return processed_count > 0
            
        except Exception as e:
            print(f"處理圖片時出錯 {image_path}: {e}")
            return False

def process_dataset(input_dirs, output_base_dir, preview=False):
    cropper = FaceCropper(preview_mode=preview)
    
    total_processed = 0
    total_success = 0
    
    for input_dir in input_dirs:
        category = os.path.basename(input_dir)
        output_dir = os.path.join(output_base_dir, category)
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
        total_files = len(image_files)
        
        for i, image_path in enumerate(image_files, 1):
            print(f"Processing {category}: {i}/{total_files} - {image_path}")
            success = cropper.process_image(image_path, output_dir)
            total_processed += 1
            if success:
                total_success += 1
                
    print(f"\n處理完成！")
    print(f"總共處理: {total_processed} 張圖片")
    print(f"成功處理: {total_success} 張圖片")
    print(f"失敗數量: {total_processed - total_success} 張圖片")

if __name__ == "__main__":
    input_dirs = [
        # "skin2_test/acne",
        # "skin2_test/dry",
        "skin2_test/normal",
        # "skin2_test/oily"
    ]
    output_base_dir = "new_skin_dataset"
    
    process_dataset(input_dirs, output_base_dir, preview=False)