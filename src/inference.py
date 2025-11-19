from ultralytics import YOLO
import cv2
import os
import argparse
import json
import yaml

def generate_grocery_list(detected_items, stock_list):
    """
    Compares detected items with stock list to generate a shopping list.
    detected_items: dict {item_name: count}
    stock_list: dict {item_name: required_count}
    """
    shopping_list = []
    print("\n--- Inventory Check ---")
    for item, required in stock_list.items():
        current = detected_items.get(item, 0)
        print(f"{item}: Found {current}/{required}")
        if current < required:
            shopping_list.append(f"{item} (Need {required - current})")
            
    return shopping_list

def load_stock_list(path):
    if not path or not os.path.exists(path):
        print("No stock list provided or found. Using default.")
        return {'apple': 5, 'milk': 1, 'eggs': 12, 'tomato': 3, 'salad_mix': 1}
    
    try:
        with open(path, 'r') as f:
            if path.endswith('.json'):
                return json.load(f)
            elif path.endswith('.yaml') or path.endswith('.yml'):
                return yaml.safe_load(f)
            else:
                # Assume simple text format: item: count
                stock = {}
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        stock[parts[0].strip()] = int(parts[1].strip())
                return stock
    except Exception as e:
        print(f"Error loading stock list: {e}. Using default.")
        return {'apple': 5, 'milk': 1, 'eggs': 12, 'tomato': 3, 'salad_mix': 1}

def run_inference(model_path, image_path, stock_list_path=None, output_dir='output'):
    """
    Runs inference on an image and generates a grocery list.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model = YOLO(model_path)
    
    # Inference
    os.makedirs(output_dir, exist_ok=True)
    results = model.predict(source=image_path, save=True, project=output_dir, name='inference', exist_ok=True, conf=0.25)
    
    # Process results
    detected_counts = {}
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
            
    print(f"\nDetected Items: {detected_counts}")

    stock_list = load_stock_list(stock_list_path)
    shopping_list = generate_grocery_list(detected_counts, stock_list)
    
    print("\n--- Grocery List ---")
    if shopping_list:
        for item in shopping_list:
            print(f"- {item}")
    else:
        print("Everything is in stock!")
        
    print(f"\nAnnotated image saved to {os.path.join(output_dir, 'inference')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to model weights')
    parser.add_argument('--stock', type=str, help='Path to stock list file (json/yaml/txt)')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    args = parser.parse_args()
    
    run_inference(args.model, args.image, args.stock, args.output)
