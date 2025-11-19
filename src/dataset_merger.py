import os
import glob
import shutil
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml

class DatasetMerger:
    def __init__(self, base_path):
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, 'data', 'raw')
        self.processed_path = os.path.join(base_path, 'data', 'processed')
        self.classes = []  # Will be populated dynamically or defined
        self.class_map = {}

    def convert_sku110k_to_yolo(self, csv_path, images_dir, output_dir):
        """
        Converts SKU110k CSV annotations to YOLO format.
        Expected CSV format: image_name, x1, y1, x2, y2, class, image_width, image_height
        """
        print(f"Processing SKU110k from {csv_path}...")
        if not os.path.exists(csv_path):
            print(f"Warning: SKU110k CSV not found at {csv_path}")
            return

        df = pd.read_csv(csv_path)
        
        # Ensure output directory exists
        labels_dir = os.path.join(output_dir, 'labels')
        images_out_dir = os.path.join(output_dir, 'images')
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(images_out_dir, exist_ok=True)

        # Group by image
        grouped = df.groupby('image_name')

        for image_name, group in tqdm(grouped, desc="Converting SKU110k"):
            image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(image_path):
                continue
            
            # Copy image
            shutil.copy(image_path, os.path.join(images_out_dir, image_name))

            # Create label file
            label_file = os.path.join(labels_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            with open(label_file, 'w') as f:
                for _, row in group.iterrows():
                    class_name = row['class']
                    if class_name not in self.class_map:
                        self.class_map[class_name] = len(self.classes)
                        self.classes.append(class_name)
                    
                    class_id = self.class_map[class_name]
                    
                    # SKU110k is usually x1, y1, x2, y2
                    x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                    w_img, h_img = row['image_width'], row['image_height']
                    
                    # Normalize to YOLO format (center_x, center_y, width, height)
                    dw = 1. / w_img
                    dh = 1. / h_img
                    
                    w = x2 - x1
                    h = y2 - y1
                    x_center = x1 + w / 2.0
                    y_center = y1 + h / 2.0
                    
                    x_center *= dw
                    w *= dw
                    y_center *= dh
                    h *= dh
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    def process_custom_fridge(self, input_dir, output_dir):
        """
        Process custom fridge data. Assumes images and corresponding .txt files in YOLO format already,
        or some other format. For now, assuming they need to be just copied or split.
        """
        print(f"Processing Custom Fridge Data from {input_dir}...")
        # Implementation depends on custom format. 
        # Placeholder: Copy files if they exist
        pass

    def create_data_yaml(self):
        yaml_content = {
            'path': self.processed_path,
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        with open(os.path.join(self.base_path, 'data', 'data.yaml'), 'w') as f:
            yaml.dump(yaml_content, f)
        print("Created data.yaml")

    def run(self):
        # Example paths - adjust based on actual raw data structure
        sku_train_csv = os.path.join(self.raw_path, 'sku110k', 'annotations', 'train.csv')
        sku_train_imgs = os.path.join(self.raw_path, 'sku110k', 'images')
        
        # Process SKU110k
        self.convert_sku110k_to_yolo(sku_train_csv, sku_train_imgs, os.path.join(self.processed_path, 'train'))
        
        # Process Custom Data (Placeholder)
        # self.process_custom_fridge(...)
        
        # Create YAML
        self.create_data_yaml()

if __name__ == "__main__":
    merger = DatasetMerger(base_path='/Users/shashank/Deep_Learning/codebase/SmartGroceryTracker')
    merger.run()
