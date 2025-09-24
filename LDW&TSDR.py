import cv2
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import os
import time
import torch
from ultralytics import YOLO
from PIL import Image
import torchvision
from ultralytics.utils import LOGGER
LOGGER.setLevel(50)  # Set logging level to CRITICAL

# Constants for meter-to-pixel conversion
ym_per_pix = 30 / 180
xm_per_pix = 3.7 / 640

# Global variables for lane tracking and warnings
prev_left_lane = None
prev_right_lane = None
left_warning_counter = 0
right_warning_counter = 0
max_warning_frames = 10
left_warning_active = False
right_warning_active = False
left_lane_counter = 0
right_lane_counter = 0
frame_threshold = 18

def RunONNX_Fixed(session, img, detection_model, classification_model, classes, detection_threshold=0.3):
    global prev_left_lane, prev_right_lane
    global left_lane_counter, right_lane_counter

    # Step 1: Resize the input image
    img_resized = cv2.resize(img, (640, 360))  # Resize to match model input
    height, width, _ = img_resized.shape

    # Step 2: Split the image into upper and lower halves
    upper_half = img_resized[:height // 2, :]
    lower_half = img_resized[height // 2:, :]

    # Detection
    results = detection_model.predict(source=upper_half, save=False, verbose=False)
    detections = results[0].boxes

    if detections is not None and len(detections) > 0:
        boxes = detections.xyxy.cpu().numpy().astype(int)

        for box in boxes:
            x1, y1, x2, y2 = box[:4]

            # Crop and preprocess ROI for classification
            roi = upper_half[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Preprocess and classify ROI
            resized = cv2.resize(roi, (224, 224))
            normalized = resized.astype(np.float32) / 255.0
            transposed = np.transpose(normalized, (2, 0, 1))
            input_tensor = np.expand_dims(transposed, axis=0)

            # Classification
            outputs = classification_model.run(None, {classification_model.get_inputs()[0].name: input_tensor})
            pred_label = classes[np.argmax(outputs[0])]

            # Annotate classification label on frame
            cv2.rectangle(upper_half, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(upper_half, pred_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)



    # Preprocess the cropped image for ONNX model
    cropped_img_preprocessed = lower_half[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and HWC to CHW
    cropped_img_preprocessed = np.ascontiguousarray(cropped_img_preprocessed, dtype=np.float32) / 255.0  # Normalize
    cropped_img_preprocessed = np.expand_dims(cropped_img_preprocessed, axis=0)  # Add batch dimension

    # Step 3: Run inference on the ONNX model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: cropped_img_preprocessed})

    # Get segmentation output
    ll_predict = np.argmax(outputs[0], axis=1)[0]  # For multi-class output
    ll_predict_resized = cv2.resize(ll_predict, (width, height // 2), interpolation=cv2.INTER_NEAREST)

    # Overlay segmented lanes onto the cropped image
    lower_half[ll_predict_resized > 0] = [0, 255, 0]  # Green for lane segments

    output_img = np.vstack((upper_half, lower_half))

    # Step 5: Continue processing for lane detection
    smoothed = cv2.GaussianBlur(ll_predict_resized.astype(np.uint8) * 255, (5, 5), 0)
    edges = cv2.Canny(smoothed, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    left_lane_points = []
    right_lane_points = []
    vehicle_x = width // 2  # Vehicle's center position
    vehicle_y = height - 1

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero

            if slope < -0.5:  # Left lane
                right_lane_points.append(((x1 + x2) // 2, (y1 + y2) // 2))
            elif slope > 0.5:  # Right lane
                left_lane_points.append(((x1 + x2) // 2, (y1 + y2) // 2))             

    # Step 6: Calculate lane centers
    left_lane_center = None
    right_lane_center = None

    if prev_left_lane is None:
        prev_left_lane = (0, 0)
    if prev_right_lane is None:
        prev_right_lane = (0, 0)

    if left_lane_points:
        left_lane_center_x = np.mean([point[0] for point in left_lane_points])
        left_lane_center_y = np.mean([point[1] for point in left_lane_points])
        left_lane_center = (left_lane_center_x, left_lane_center_y)
    if right_lane_points:
        right_lane_center_x = np.mean([point[0] for point in right_lane_points])
        right_lane_center_y = np.mean([point[1] for point in right_lane_points])
        right_lane_center = (right_lane_center_x, right_lane_center_y)

    # Apply Exponential Moving Average (EMA) to smooth lane center tracking
    alpha = 0.7  # Smoothing factor
    if left_lane_center:
        left_lane_center = (
            alpha * left_lane_center[0] + (1 - alpha) * prev_left_lane[0],
            alpha * left_lane_center[1] + (1 - alpha) * prev_left_lane[1],
        )
        left_lane_counter = 0
    else:
        left_lane_center = prev_left_lane
        left_lane_counter += 1

    if right_lane_center:
        right_lane_center = (
            alpha * right_lane_center[0] + (1 - alpha) * prev_right_lane[0],
            alpha * right_lane_center[1] + (1 - alpha) * prev_right_lane[1],
        )
        right_lane_counter = 0
    else:
        right_lane_center = prev_right_lane
        right_lane_counter += 1

    # Step 7: Reset lane center if unchanged for 40 frames
    if left_lane_counter >= frame_threshold:
        left_lane_center = None
        prev_left_lane = (0, 0)
        left_lane_counter = 0

    if right_lane_counter >= frame_threshold:
        right_lane_center = None
        prev_right_lane = (0, 0)
        right_lane_counter = 0

    # Update global variables
    prev_left_lane = left_lane_center if left_lane_center else prev_left_lane
    prev_right_lane = right_lane_center if right_lane_center else prev_right_lane

    # Step 8: Calculate distances and generate warnings
    warning = "OK"
    left_warning_distance = 80  # Threshold for left lane departure in pixels
    right_warning_distance = 80  # Threshold for right lane departure in pixels
    close_window_height = 100
    close_window_top = height - close_window_height

    # Check left lane departure
    if left_lane_center and (close_window_top <= left_lane_center[1] + height // 2 <= height - 1):
        horizontal_left_distance = abs(vehicle_x - left_lane_center[0])
        if horizontal_left_distance < left_warning_distance:
            warning = "Warning: Too Close to right Lane!"

    # Check right lane departure
    if right_lane_center and (close_window_top <= right_lane_center[1] + height // 2 <= height - 1):
        horizontal_right_distance = abs(vehicle_x - right_lane_center[0])
        if horizontal_right_distance < right_warning_distance:
            warning = "Warning: Too Close to left Lane!"

    # Step 9: Visualize results
    if left_lane_center:
        cv2.circle(output_img, (int(left_lane_center[0]), int(left_lane_center[1]) + height // 2), 5, (255, 0, 0), -1)
    if right_lane_center:
        cv2.circle(output_img, (int(right_lane_center[0]), int(right_lane_center[1]) + height // 2), 5, (0, 0, 255), -1)
    cv2.circle(output_img, (vehicle_x, vehicle_y), 5, (0, 255, 255), -1)
    cv2.putText(output_img, warning, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw safe zone thresholds
    cv2.line(output_img, (vehicle_x - left_warning_distance, height - 1),
             (vehicle_x - left_warning_distance, close_window_top), (255, 255, 255), 2)
    cv2.line(output_img, (vehicle_x + right_warning_distance, height - 1),
             (vehicle_x + right_warning_distance, close_window_top), (255, 255, 255), 2)

    # Draw close window
    cv2.rectangle(output_img, (vehicle_x - left_warning_distance, close_window_top),
                  (vehicle_x + right_warning_distance, height - 1), (0, 255, 0), 2)

    return output_img

def load_detection_model_yolo(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return YOLO(model_path, verbose=False)

def load_onnx_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    return ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

def process_video(input_video_path, output_video_path, detection_model, classification_model, classes):
    cap = cv2.VideoCapture(input_video_path)
    session = load_onnx_model("Twin(640x180).onnx")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for _ in tqdm(range(frame_count), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            processed_frame = RunONNX_Fixed(session, frame, detection_model, 
                                           classification_model, classes)

            processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))                               
            out.write(processed_frame)
        except Exception as e:
            print(f"Error processing frame {_}: {e}")
    
    cap.release()
    out.release()

def main():
    input_video_path = 'input video path'
    output_video_path = 'output video path'
    
    detection_model = load_detection_model_yolo("path to yolov8.onnx")  
    classification_model = load_onnx_model("path to mobilenetv3.onnx")
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
    
    process_video(input_video_path, output_video_path, detection_model,
                classification_model, classes)

if __name__ == "__main__":
    main()




