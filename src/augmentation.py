import albumentations as A
import cv2
import numpy as np
import random
import os
import argparse

class AugmentationPipeline:
    def __init__(self, blur_limit=(3, 7), noise_var=(10.0, 50.0)):
        self.transform = A.Compose([
            A.Resize(height=640, width=640),
            A.Rotate(limit=15, p=0.5),
            A.GaussianBlur(blur_limit=blur_limit, p=0.5), # Increased p for visibility
            A.GaussNoise(var_limit=noise_var, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def augment(self, image, bboxes, class_labels):
        """
        Applies augmentation to an image and its bounding boxes.
        bboxes: List of [x_center, y_center, width, height]
        class_labels: List of class IDs
        """
        try:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            return transformed['image'], transformed['bboxes'], transformed['class_labels']
        except Exception as e:
            print(f"Augmentation failed: {e}")
            return image, bboxes, class_labels

    def visualize(self, image, bboxes, class_labels, save_path=None):
        """
        Visualizes bounding boxes on the image.
        """
        img_vis = image.copy()
        h, w = img_vis.shape[:2]
        
        for bbox, label in zip(bboxes, class_labels):
            xc, yc, bw, bh = bbox
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_vis, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        if save_path:
            cv2.imwrite(save_path, img_vis)
        return img_vis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to input image for testing')
    parser.add_argument('--output', type=str, default='augmented_test.jpg', help='Path to save output')
    args = parser.parse_args()

    pipeline = AugmentationPipeline()
    
    if args.image and os.path.exists(args.image):
        img = cv2.imread(args.image)
        # Dummy boxes for visualization test if no label file provided
        # Assuming whole image is the object for demo
        bboxes = [[0.5, 0.5, 0.5, 0.5]] 
        labels = [0]
        
        aug_img, aug_boxes, aug_labels = pipeline.augment(img, bboxes, labels)
        pipeline.visualize(aug_img, aug_boxes, aug_labels, save_path=args.output)
        print(f"Saved augmented image to {args.output}")
    else:
        print("Augmentation Pipeline Initialized. Provide --image to test.")
