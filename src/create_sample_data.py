import cv2
import numpy as np
import os
import random
from tqdm import tqdm

class SampleDataGenerator:
    def __init__(self, base_path, num_images=50):
        self.base_path = base_path
        self.num_images = num_images
        self.classes = ['apple', 'milk', 'tomato', 'salad_mix']
        self.colors = {
            'apple': (0, 0, 255),      # Red
            'milk': (255, 255, 255),   # White
            'tomato': (0, 0, 200),     # Darker Red
            'salad_mix': (0, 255, 0)   # Green
        }
        self.output_dir = os.path.join(base_path, 'data', 'raw', 'sample_dataset')
        
    def create_directories(self):
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        # We will generate labels directly in YOLO format for simplicity in this sample generator,
        # or we can generate a CSV to test the merger. Let's generate a CSV to test the merger logic.
        os.makedirs(os.path.join(self.output_dir, 'annotations'), exist_ok=True)

    def generate_image(self, img_id):
        # Create a "fridge" background (light grey/white with some noise)
        img = np.full((640, 640, 3), 240, dtype=np.uint8)
        
        # Add some random noise to background
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        annotations = []
        
        # Place random items
        num_items = random.randint(3, 10)
        for _ in range(num_items):
            cls = random.choice(self.classes)
            color = self.colors[cls]
            
            # Random position and size
            w = random.randint(50, 150)
            h = random.randint(50, 150)
            x = random.randint(0, 640 - w)
            y = random.randint(0, 640 - h)
            
            # Draw item
            if cls in ['apple', 'tomato']:
                # Draw circle
                center = (x + w//2, y + h//2)
                radius = min(w, h) // 2
                cv2.circle(img, center, radius, color, -1)
            else:
                # Draw rectangle
                cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
                
            # Add "occlusion" by drawing another item on top? 
            # Simple painter's algorithm here.
            
            annotations.append({
                'image_name': f"sample_{img_id}.jpg",
                'x1': x,
                'y1': y,
                'x2': x + w,
                'y2': y + h,
                'class': cls,
                'image_width': 640,
                'image_height': 640
            })
            
        return img, annotations

    def run(self):
        self.create_directories()
        all_annotations = []
        
        print(f"Generating {self.num_images} sample images...")
        for i in tqdm(range(self.num_images)):
            img, anns = self.generate_image(i)
            cv2.imwrite(os.path.join(self.output_dir, 'images', f"sample_{i}.jpg"), img)
            all_annotations.extend(anns)
            
        # Save annotations to CSV (mimicking SKU110k format)
        import pandas as pd
        df = pd.DataFrame(all_annotations)
        df.to_csv(os.path.join(self.output_dir, 'annotations', 'annotations.csv'), index=False)
        print(f"Sample dataset created at {self.output_dir}")

if __name__ == "__main__":
    generator = SampleDataGenerator(base_path='/Users/shashank/Deep_Learning/codebase/SmartGroceryTracker')
    generator.run()
