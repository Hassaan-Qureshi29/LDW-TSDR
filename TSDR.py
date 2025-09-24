import os
import time
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import timm
import torch.nn as nn
import joblib
import numpy as np
import cv2
from ultralytics import YOLO

# ----------------- Detection Model Functions -----------------

# Load the YOLOv8 detection model
def load_detection_model_yolo(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Detection model not found at {model_path}")
    return YOLO(model_path)  # Load the model using Ultralytics YOLO

def preprocess_detection_image(image_path):
    """Loads an image, converts it to RGB, resizes it to 640x360, and returns it."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    #image = image.resize((640, 360))  # Resize image to 640x360
    
    return image


# Perform inference with the YOLOv8 model
def detect_traffic_signs_yolo(model, image, threshold=0.5):
    results = model(image)  # Run inference directly
    detections = results[0].boxes.data.cpu().numpy()  # Bounding boxes with confidence and class
    filtered_results = [
        {
            "box": [float(det[0]), float(det[1]), float(det[2]), float(det[3])],  # [x1, y1, x2, y2]
            "score": float(det[4]),  # Confidence score
        }
        for det in detections if det[4] >= threshold  # Confidence threshold
    ]
    return filtered_results

# ----------------- Classification Model Functions -----------------
def load_classification_model(model_path, num_classes):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classification model not found at {model_path}")
    
    # Create model with same architecture modifications
    model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=num_classes)
    
    # Add dropout layer to match training architecture
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        model.classifier
    )
    
    # Load weights with strict=False to handle any remaining mismatches
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model

# Preprocess a cropped bounding box image for the classification model
def preprocess_classification_image(cropped_image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    return transform(cropped_image).unsqueeze(0)

# Perform classification
def classify_traffic_sign(model, image_tensor, classes):
    with torch.no_grad():
        outputs = model(image_tensor)  # Run inference
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        return class_idx, classes[class_idx]

# ----------------- Main Functionality -----------------

if __name__ == "__main__":
    detection_model_path = "weights\Yolov8s.pt"  # Replace with your YOLOv8 model path
    classification_model_path = "Weights\mobilenetv3_v2.pth"  # classification model path
    input_folder = "input folder path"  # Folder containing images
    output_folder = "output folder path"  # Folder to save images with predictions
    detection_threshold = 0.3  # Detection confidence threshold
    num_classes = 35  # Number of classes for classification

    # Traffic sign classes
    classes = [
        "Bridge Ahead", "Cross Roads", "Give Way", "Left bend", "No Horns", 
        "No Mobile Allowed", "No Overtaking", "No Parking", "No U-Turn", 
        "No left turn", "No right turn", "Parking", "Pedestrians", 
        "Railway Crossing", "Right bend", "Road Divides", "Roundabout Ahead", 
        "Sharp Right Turn", "Slow", "Speed Breaker Ahead", 
        "Speed Limit (20 kmph)", "Speed Limit (25 kmph)", "Speed Limit (30 kmph)", 
        "Speed Limit (40 kmph)", "Speed Limit (45 kmph)", "Speed Limit (50 kmph)", 
        "Speed Limit (60 kmph)", "Speed Limit (65 kmph)", "Speed Limit (70 kmph)", 
        "Speed Limit (80 kmph)", "Steep Descent", "Stop 1", "Stop 2", 
        "U-Turn", "Zigzag Road Ahead"
    ]

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Load models
        print("Loading models...")
        detection_model = load_detection_model_yolo(detection_model_path)
        classification_model = load_classification_model(classification_model_path, num_classes)

        # Process each image in the input folder
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")]
        if not image_files:
            print("No JPEG images found in the input folder.")
            exit()

        total_time = 0
        for image_file in image_files:
            print(f"Processing {image_file}...")
            image_path = os.path.join(input_folder, image_file)
            original_image = preprocess_detection_image(image_path)

            # Start time for FPS calculation
            start_time = time.time()

            # Run detection
            detections = detect_traffic_signs_yolo(detection_model, original_image, threshold=detection_threshold)

            # Visualize and classify each detected traffic sign
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(original_image)

            for det in detections:
                box = det["box"]
                confidence = det["score"]

                # Ensure bounding box values are within image dimensions
                width, height = original_image.size
                box = [
                    max(0, min(box[0], width)),
                    max(0, min(box[1], height)),
                    max(0, min(box[2], width)),
                    max(0, min(box[3], height))
                ]

                # Crop the detected box from the original image
                cropped_image = original_image.crop((box[0], box[1], box[2], box[3]))

                # Preprocess and classify the cropped image
                cropped_tensor = preprocess_classification_image(cropped_image)
                class_idx, predicted_class = classify_traffic_sign(classification_model, cropped_tensor, classes)

                # Draw bounding box and label
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    box[0],
                    box[1] - 10,
                    f"{predicted_class} ({confidence:.2f})",
                    color="red",
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.7),
                )

            # Save the image with predictions
            output_path = os.path.join(output_folder, image_file)
            plt.axis("off")
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            # End time for FPS calculation
            end_time = time.time()
            total_time += (end_time - start_time)

        # Calculate and print FPS
        fps = len(image_files) / total_time
        print(f"Processed {len(image_files)} images with an average FPS: {fps:.2f}")

    except Exception as e:
        print(f"Error: {e}")
