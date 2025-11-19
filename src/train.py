from ultralytics import YOLO
import os
import yaml
import argparse

def train_model(data_yaml_path, epochs=100, img_size=640, batch_size=16, project_name='SmartGroceryTracker'):
    """
    Trains the YOLOv8 model.
    """
    # Load a model
    model = YOLO('yolov8n.pt') 

    print(f"Starting training with {data_yaml_path}...")
    
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project_name,
        name='yolov8_custom',
        exist_ok=True,
        # Specific constraints from paper/requirements:
        box=7.5,  # box loss gain
        cls=0.5,  # cls loss gain
        dfl=1.5,  # dfl loss gain
        iou=0.7,  # IoU threshold for NMS (handling occlusion)
        val=True, # Enable validation during training
        plots=True, # Save plots
    )
    
    print("Training completed.")
    print(f"Results saved to {results.save_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/Users/shashank/Deep_Learning/codebase/SmartGroceryTracker/data/data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: {args.data} not found. Please run dataset_merger.py first.")
    else:
        train_model(args.data, epochs=args.epochs, batch_size=args.batch)
