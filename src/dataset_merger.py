import os
import glob
import shutil
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
import argparse

class DatasetMerger:
    def __init__(self, base_path):
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, 'data', 'raw')
        self.processed_path = os.path.join(base_path, 'data', 'processed')
        self.classes = []
        self.class_map = {}

    def convert_csv_to_yolo(self, csv_path, images_dir, output_dir, dataset_name="dataset"):
        """
        Generic CSV to YOLO converter.
        Expected CSV columns: image_name, x1, y1, x2, y2, class, image_width, image_height
        """
        print(f"Processing {dataset_name} from {csv_path}...")
        if not os.path.exists(csv_path):
            print(f"Warning: CSV not found at {csv_path}")
            return

        df = pd.read_csv(csv_path)
        
        labels_dir = os.path.join(output_dir, 'labels')
        images_out_dir = os.path.join(output_dir, 'images')
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(images_out_dir, exist_ok=True)

        grouped = df.groupby('image_name')

        for image_name, group in tqdm(grouped, desc=f"Converting {dataset_name}"):
            image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(image_path):
                # Try checking if image_name has path info or if it's just filename
                # For sample dataset, it's flat.
                continue
            
            target_image_path = os.path.join(images_out_dir, image_name)
            if not os.path.exists(target_image_path):
                shutil.copy(image_path, target_image_path)

            label_file = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')
            
            with open(label_file, 'w') as f:
                for _, row in group.iterrows():
                    class_name = row['class']
                    if class_name not in self.class_map:
                        self.class_map[class_name] = len(self.classes)
                        self.classes.append(class_name)
                    
                    class_id = self.class_map[class_name]
                    
                    x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                    w_img, h_img = row['image_width'], row['image_height']
                    
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

    def create_data_yaml(self):
        yaml_content = {
            'path': self.processed_path,
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images', # Optional
            'nc': len(self.classes),
            'names': self.classes
        }
        
        with open(os.path.join(self.base_path, 'data', 'data.yaml'), 'w') as f:
            yaml.dump(yaml_content, f)
        print(f"Created data.yaml with classes: {self.classes}")

    def run(self, use_sample=False):
        if use_sample:
            print("Using Sample Dataset...")
            sample_csv = os.path.join(self.raw_path, 'sample_dataset', 'annotations', 'annotations.csv')
            sample_imgs = os.path.join(self.raw_path, 'sample_dataset', 'images')
            
            # Split into train/val (simple file split logic or just process all to train for demo)
            # Ideally we split the CSV. For now, let's put everything in train and a subset in val.
            self.convert_csv_to_yolo(sample_csv, sample_imgs, os.path.join(self.processed_path, 'train'), "Sample Train")
            self.convert_csv_to_yolo(sample_csv, sample_imgs, os.path.join(self.processed_path, 'val'), "Sample Val") # Duplicate for demo
            
        else:
            # SKU110k
            sku_train_csv = os.path.join(self.raw_path, 'sku110k', 'annotations', 'train.csv')
            sku_train_imgs = os.path.join(self.raw_path, 'sku110k', 'images')
            self.convert_csv_to_yolo(sku_train_csv, sku_train_imgs, os.path.join(self.processed_path, 'train'), "SKU110k Train")
            
            # Add validation split logic here for full dataset
        
        self.create_data_yaml()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/Users/shashank/Deep_Learning/codebase/SmartGroceryTracker')
    parser.add_argument('--sample', action='store_true', help='Use sample dataset')
    args = parser.parse_args()

    merger = DatasetMerger(base_path=args.base_path)
    merger.run(use_sample=args.sample)
