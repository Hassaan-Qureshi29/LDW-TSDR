import os
import time
import numpy as np
import cv2
import torch
import timm
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO
import onnxruntime as ort
from ultralytics.utils import LOGGER
LOGGER.setLevel(50)  # Set logging level to CRITICAL
from scipy.optimize import linear_sum_assignment  # Add this import

# Put these imports and globals at top of your file (once)
import os
import csv
import datetime

LOG_FILE = "detection_log.csv"

# Initialize counters (persist across frames)
total_frames = 0
detection_frames = 0
model_pred_frames = 0
kalman_pred_frames = 0

# Create CSV with header if not exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "frame_num",
            "detection_present", "recognition_name",
            "model_predicted", "kalman_predicted"
        ])



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

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.hits = 1
        self.time_since_update = 0
        self.age = 0

        self.label = None
        self.label_conf = 0.0

        # Use OpenCV KalmanFilter (you already have cv2)
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)

        self.kf.transitionMatrix = np.array([[1, 0, 0, 0, 1, 0, 0],
                                             [0, 1, 0, 0, 0, 1, 0],
                                             [0, 0, 1, 0, 0, 0, 1],
                                             [0, 0, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0],
                                             [0, 0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 0, 0, 0, 1]], np.float32)

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0]], np.float32)

    def update(self, bbox, label=None, conf=0.0):
        self.time_since_update = 0
        self.hits += 1
        self.age += 1

        if label is not None and conf >= CLASSIFICATION_CONFIDENCE:
            self.label = label
            self.label_conf = conf

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        meas = np.array([[cx], [cy], [w], [h]], np.float32)
        self.kf.correct(meas)

    def predict(self):
        state = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return state

    def get_state(self):
        state = self.kf.statePost.flatten()
        cx, cy, w, h = state[0], state[1], state[2], state[3]
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))

    # Compute IoU matrix
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = compute_iou(det[:4], trk[:4])

    # Hungarian algorithm (maximize IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    # Create matched indices as 2D array
    matched_indices = np.array(list(zip(row_ind, col_ind)))

    # Filter out low IoU matches
    if matched_indices.size > 0:
        matches = iou_matrix[matched_indices[:, 0], matched_indices[:, 1]] > iou_threshold
        matched_indices = matched_indices[matches]
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    # Unmatched detections
    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    # Unmatched trackers
    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    return matched_indices, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort:
    def __init__(self, max_age=8, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = []
        for t in self.trackers:
            pred = t.predict()
            trks.append(t.get_state())

        trks = np.array(trks) if trks else np.empty((0, 4))

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)

        # Output active tracks
        active = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            d = trk.get_state()
            if (trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                active.append(trk)
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return active
    
def compute_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou    
# ----------------- USER CONFIG -----------------
PTH_CLASSIFIER_PATH = 'mobilenetv3_v2.pth'   # <-- update
YOLO_MODEL_PATH = 'yolov8s.onnx'             # <-- update (you had this)
TWINLITE_ONNX_PATH = 'TQ.onnx'               # <-- update
VIDEO_PATH = "Test.mp4"
OUTPUT_FILE = "output_processed.avi"
WRITE_OUTPUT = True
SHOW_WINDOW = True
CAM_WIDTH = 640
CAM_HEIGHT = 360
CAM_FPS = 20
DETECTION_CONFIDENCE = 0.8
CLASSIFICATION_CONFIDENCE = 0.7
frame_num = 0
mot_tracker = None
# ------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Example classes (use your actual class list)
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

def load_classification_model(model_path, num_classes, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classification model not found at {model_path}")

    # Create model with same architecture used in training
    model = timm.create_model(
        "mobilenetv3_small_100",
        pretrained=True,
        num_classes=num_classes
    )

    # Add dropout to match training architecture
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        model.classifier
    )

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    print("MobileNetV3 classifier loaded successfully.")
    return model

# ----------------- Preproc for classifier -----------------
classifier_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard imagenet
                         std=[0.229, 0.224, 0.225])
])

def classify_roi_torch(model, roi_bgr):
    """
    roi_bgr: OpenCV BGR numpy array
    returns: predicted label (string) and confidence float
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None, 0.0

    # Convert BGR->RGB
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    tensor = classifier_transform(roi_rgb).unsqueeze(0).to(device)  # shape 1,C,H,W

    model.eval()
    with torch.no_grad():
        out = model(tensor)
        if out is None:
            return None, 0.0
        if isinstance(out, (list, tuple)):
            out = out[0]
        probs = F.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)
        label = classes[int(idx.item())] if 0 <= int(idx.item()) < len(classes) else str(int(idx.item()))
        return label, float(conf.item())

# ----------------- ONNX loader -----------------
def load_onnx_session(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX not found: {path}")
    return ort.InferenceSession(path, providers=['CPUExecutionProvider'])

# ----------------- Main processing function (per-frame) -----------------
def Run_frame(session_twin, frame_bgr, detection_model, classifier_model, mot_tracker):
    global frame_num
    global prev_left_lane, prev_right_lane
    global left_lane_counter, right_lane_counter

    img_resized = cv2.resize(frame_bgr, (CAM_WIDTH, CAM_HEIGHT))
    h, w, _ = img_resized.shape

    upper_half = img_resized[:h//2, :].copy()
    lower_half = img_resized[h//2:, :].copy()

    # -------------------------------------------------
    # YOLO Detection (UNCHANGED)
    # -------------------------------------------------
    detections = []
    try:
        results = detection_model.predict(
            source=upper_half, save=False, show=False, verbose=False,
            conf=DETECTION_CONFIDENCE, iou=0.45
        )
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                detections.append([*box, float(confs[i])])
    except:
        pass

    tracked_objects = mot_tracker.update(
        np.array(detections) if detections else np.empty((0, 5))
    )

    for track in tracked_objects:
        x1, y1, x2, y2 = map(int, track.get_state())
        y2 = min(y2, h // 2)

        if track.time_since_update == 0:
            roi = upper_half[y1:y2, x1:x2]
            if roi.size > 0:
                label, score = classify_roi_torch(classifier_model, roi)
                if score >= CLASSIFICATION_CONFIDENCE:
                    track.label = label
                    track.label_conf = score

        label = track.label if track.label else "Sign"
        conf = track.label_conf if track.label_conf else 0.0

        cv2.rectangle(upper_half, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            upper_half, f"{label} ({conf:.2f})",
            (x1, max(10, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

    # -------------------------------------------------
    # Lane Segmentation (UNCHANGED)
    # -------------------------------------------------
    inp = lower_half[:, :, ::-1].transpose(2, 0, 1)
    inp = np.expand_dims(inp, 0).astype(np.float32) / 255.0

    outputs = session_twin.run(
        [session_twin.get_outputs()[0].name],
        {session_twin.get_inputs()[0].name: inp}
    )

    ll_predict = np.argmax(outputs[0], axis=1)[0]
    ll_predict_resized = cv2.resize(
        ll_predict, (w, h // 2), interpolation=cv2.INTER_NEAREST
    )

    # -------------------------------------------------
    # 🔥 NEW: DIRECTIONAL LANE MASK EXPANSION 🔥
    # -------------------------------------------------
    lane_bin = (ll_predict_resized > 0).astype(np.uint8) * 255
    hh, ww = lane_bin.shape

    expanded = lane_bin.copy()

    # Get line points for fitting
    smoothed = cv2.GaussianBlur(lane_bin, (5, 5), 0)
    edges = cv2.Canny(smoothed, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=50, minLineLength=50, maxLineGap=10
    )

    left_points = []
    right_points = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope > 0.5:
                left_points.append((x1, y1))
                left_points.append((x2, y2))
            elif slope < -0.5:
                right_points.append((x1, y1))
                right_points.append((x2, y2))

    # Function to extend mask for one side
    def extend_mask(points, expanded):
        if not points:
            return
        points_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        [vx, vy, x0, y0] = cv2.fitLine(points_np, cv2.DIST_L2, 0, 0.01, 0.01)
        # Ensure direction is downward (vy > 0)
        if vy < 0:
            vx = -vx
            vy = -vy
        # Compute ts
        ts = []
        for px, py in points:
            t = (px - x0) * vx + (py - y0) * vy
            ts.append(t)
        max_t = max(ts)
        # Point at max_t
        x_max = x0 + max_t * vx
        y_max = y0 + max_t * vy
        # t for bottom
        if vy == 0:
            return
        t_bot = (hh - 1 - y0) / vy
        if t_bot <= max_t:
            return  # No need to extend downward
        # Point at bottom
        x_bot = x0 + t_bot * vx
        y_bot = hh - 1
        # Thickness
        thickness = 10  # Adjust based on typical lane mask width if needed
        # Draw thick line to expand mask
        cv2.line(expanded, (int(x_max), int(y_max)), (int(x_bot), int(y_bot)), 255, thickness)

    # Extend left and right
    extend_mask(left_points, expanded)
    extend_mask(right_points, expanded)

    # Optional: small morphological closing (safe)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    expanded = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, kernel)

    # -------------------------------------------------
    # Visualization (UNCHANGED)
    # -------------------------------------------------
    lower_half[expanded > 0] = [0, 255, 0]
    output_img = np.vstack((upper_half, lower_half))

    # -------------------------------------------------
    # Lane Detection (CANNY → HOUGH → SLOPE) UNCHANGED
    # -------------------------------------------------
    smoothed = cv2.GaussianBlur(expanded, (5, 5), 0)
    edges = cv2.Canny(smoothed, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=50, minLineLength=50, maxLineGap=10
    )

    left_lane_points = []
    right_lane_points = []

    vehicle_x = w // 2
    vehicle_y = h - 1

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if slope < -0.5:
                right_lane_points.append(((x1 + x2) // 2, (y1 + y2) // 2))
            elif slope > 0.5:
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
    close_window_top = h - close_window_height

    # Check left lane departure
    if left_lane_center and (close_window_top <= left_lane_center[1] + h // 2 <= h - 1):
        horizontal_left_distance = abs(vehicle_x - left_lane_center[0])
        if horizontal_left_distance < left_warning_distance:
            warning = "Warning: Too Close to right Lane!"

    # Check right lane departure
    if right_lane_center and (close_window_top <= right_lane_center[1] + h // 2 <= h - 1):
        horizontal_right_distance = abs(vehicle_x - right_lane_center[0])
        if horizontal_right_distance < right_warning_distance:
            warning = "Warning: Too Close to left Lane!"

    # Step 9: Visualize results
    if left_lane_center:
        cv2.circle(output_img, (int(left_lane_center[0]), int(left_lane_center[1]) + h // 2), 5, (255, 0, 0), -1)
    if right_lane_center:
        cv2.circle(output_img, (int(right_lane_center[0]), int(right_lane_center[1]) + h // 2), 5, (0, 0, 255), -1)
    cv2.circle(output_img, (vehicle_x, vehicle_y), 5, (0, 255, 255), -1)
    cv2.putText(output_img, warning, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Draw safe zone thresholds
    cv2.line(output_img, (vehicle_x - left_warning_distance, h - 1),
             (vehicle_x - left_warning_distance, close_window_top), (255, 255, 255), 2)
    cv2.line(output_img, (vehicle_x + right_warning_distance, h - 1),
             (vehicle_x + right_warning_distance, close_window_top), (255, 255, 255), 2)

    # Draw close window
    cv2.rectangle(output_img, (vehicle_x - left_warning_distance, close_window_top),
                  (vehicle_x + right_warning_distance, h - 1), (0, 255, 0), 2)

    return output_img


# ----------------- RealSense streaming and main -----------------
def main():
    print("Loading models...")
    classifier_model = load_classification_model(
        PTH_CLASSIFIER_PATH,
        num_classes=len(classes),
        device=device
    )

    print("Classifier loaded.")

    detection_model = YOLO(YOLO_MODEL_PATH)
    print("YOLO loaded.")

    session_twin = load_onnx_session(TWINLITE_ONNX_PATH)
    print("TwinLite ONNX loaded.")

    global mot_tracker
    mot_tracker = Sort(max_age=10, min_hits=2)  # Tune: max_age=8 means persist ~8 missed frames
    # mot_tracker = Sort(max_age=10, min_hits=2, iou_threshold=0.3)

    # ---------------- VIDEO INPUT ----------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {frame_width}x{frame_height}, FPS={fps}, Frames={frame_count}")

    # ---------------- VIDEO OUTPUT ----------------
    writer = None
    if WRITE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(
            OUTPUT_FILE,
            fourcc,
            CAM_FPS,
            (CAM_WIDTH, CAM_HEIGHT)   
        )

    # ---------------- PROCESS LOOP ----------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ RESIZE FIRST (IMPORTANT)
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

        t0 = time.time()
        out_img = Run_frame(session_twin, frame, detection_model, classifier_model, mot_tracker)
        t1 = time.time()

        fps_disp = 1.0 / (t1 - t0 + 1e-6)
        cv2.putText(
            out_img,
            f"FPS: {fps_disp:.1f}",
            (10, CAM_HEIGHT - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        if SHOW_WINDOW:
            cv2.imshow("Video LDW + TSD", out_img)

        if WRITE_OUTPUT and writer is not None:
            writer.write(out_img)   # ✅ WRITE RESIZED FRAME

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Processing finished.")


if __name__ == "__main__":
    main()