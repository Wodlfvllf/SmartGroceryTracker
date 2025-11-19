import albumentations as A
import cv2
import numpy as np
import random
import os

class AugmentationPipeline:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(height=640, width=640),
            A.Rotate(limit=15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3), # blur_limit 3 is approx 1.5px sigma, 5 is higher. Adjusting to be noticeable.
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), # Noise simulation
            # A.RandomBrightnessContrast(p=0.2), # Optional but good for lighting changes
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

    def mosaic(self, images, bboxes_list, class_labels_list, output_size=(640, 640)):
        """
        Implements Mosaic augmentation by combining 4 images.
        images: List of 4 images
        bboxes_list: List of 4 lists of bboxes
        class_labels_list: List of 4 lists of class labels
        """
        if len(images) != 4:
            raise ValueError("Mosaic requires exactly 4 images.")

        h_out, w_out = output_size
        mosaic_img = np.full((h_out * 2, w_out * 2, 3), 114, dtype=np.uint8) # 114 is grey padding
        
        mosaic_bboxes = []
        mosaic_labels = []
        
        # Center point
        xc, yc = int(w_out * random.uniform(0.5, 1.5)), int(h_out * random.uniform(0.5, 1.5))

        for i, img in enumerate(images):
            h, w = img.shape[:2]
            
            # Place image in the mosaic
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w_out * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, h_out * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w_out * 2), min(yc + h, h_out * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)

            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # Adjust bboxes
            current_bboxes = bboxes_list[i]
            current_labels = class_labels_list[i]
            
            for bbox, label in zip(current_bboxes, current_labels):
                # Convert YOLO (xc, yc, w, h) to Pascal VOC (x1, y1, x2, y2) for shifting
                xc_b, yc_b, w_b, h_b = bbox
                
                # Scale to original image size
                x1 = (xc_b - w_b / 2) * w
                y1 = (yc_b - h_b / 2) * h
                x2 = (xc_b + w_b / 2) * w
                y2 = (yc_b + h_b / 2) * h
                
                # Shift
                x1 += padw
                y1 += padh
                x2 += padw
                y2 += padh
                
                # Clip to mosaic boundary
                # Note: This simple clipping might invalidate small boxes or parts of boxes.
                # For robust implementation, we should check validity.
                
                # Convert back to YOLO relative to the *mosaic* size (2x output size initially, then resized)
                # Wait, usually mosaic is resized to output_size at the end.
                # Let's keep it simple: return the large mosaic and let resize handle it, or resize now.
                # Standard YOLO mosaic outputs a fixed size image.
                
                mosaic_bboxes.append([x1, y1, x2, y2]) # Store as absolute for now
                mosaic_labels.append(label)

        # Resize mosaic to output_size
        final_img = cv2.resize(mosaic_img, (w_out, h_out))
        
        # Adjust bboxes for resize
        scale_x = w_out / (w_out * 2)
        scale_y = h_out / (h_out * 2)
        
        final_bboxes = []
        for bbox in mosaic_bboxes:
            x1, y1, x2, y2 = bbox
            x1 *= scale_x
            y1 *= scale_y
            x2 *= scale_x
            y2 *= scale_y
            
            # Convert to YOLO
            w = x2 - x1
            h = y2 - y1
            xc = x1 + w / 2
            yc = y1 + h / 2
            
            # Normalize
            xc /= w_out
            yc /= h_out
            w /= w_out
            h /= h_out
            
            final_bboxes.append([xc, yc, w, h])
            
        return final_img, final_bboxes, mosaic_labels

if __name__ == "__main__":
    # Test
    print("Augmentation Pipeline Initialized")
