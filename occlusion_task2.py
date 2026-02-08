import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os  # Added to check if file exists

# --- Helper Function: Calculate Intersection over Union (IoU) ---


def calculate_iou(boxA, boxB):
    """Calculates the Intersection over Union (IoU) between two bounding boxes."""
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# --- 1. Setup Models and Device ---
print("Loading models... This may take a moment.")
# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLOv8n (Object Detector)
# This will automatically download the model if not present.
model_yolo = YOLO('yolov8n.pt')
model_yolo.to(device)

# Load MiDaS Small (Depth Estimation Model)
# Using torch.hub to get the lightweight "MiDaS_small" model
model_midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model_midas.to(device)
model_midas.eval()  # Set model to evaluation mode

# Load MiDaS transforms
midas_transforms = torch.hub.load(
    "intel-isl/MiDaS", "transforms").small_transform

print("Models loaded successfully.")

# --- 2. Load Image (Local File Version) ---
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# >> SET YOUR IMAGE FILE NAME HERE <<
# <-- e.g., 'car_image.png', 'test.jpg'
image_filename = 'Traffic_Road_Image_5.jpg'
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
print(f"Loading image from local file: {image_filename}")

if not os.path.exists(image_filename):
    print(f"FATAL ERROR: Image file not found at '{image_filename}'")
    print(f"Please make sure the image is in the same folder as the script.")
    exit()

try:
    # Read the image with OpenCV (for YOLO and visualization) - BGR format
    img_cv = cv2.imread(image_filename)
    if img_cv is None:
        raise Exception(
            f"cv2.imread failed to load the image. It might be corrupted or in an unsupported format.")

    # *** THIS IS THE FIX ***
    # Create an RGB version (as a NumPy array) for MiDaS
    # MiDaS transforms expect an RGB NumPy array, not a PIL Image or BGR array
    img_rgb_np = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    img_h, img_w = img_cv.shape[:2]
    print(f"Image '{image_filename}' loaded successfully ({img_w}x{img_h}).")

except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# --- 3. Task 1: Run YOLOv8 ---
# We only care about vehicle classes (COCO indices: 2=car, 3=motorcycle, 5=bus, 7=truck)
vehicle_classes = [2, 3, 5, 7]
print("Running YOLOv8 detection...")
# Use img_cv (BGR) as YOLO was trained on it
results = model_yolo.predict(
    img_cv, classes=vehicle_classes, conf=0.4, device=device, verbose=False)

# Store detections in a list of dictionaries
detections = []
for r in results:
    for box in r.boxes:
        detections.append({
            'bbox': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
            # Get class index as an integer
            'cls': int(box.cls[0].cpu().item()),
            'conf': box.conf[0].cpu().item(),
            'occluded': False,  # Initialize as not occluded
            'median_depth': 0.0
        })
print(f"Found {len(detections)} vehicles.")

# --- 4. Task 2: Generate Depth Map ---
print("Generating depth map...")

# *** THIS IS THE SECOND PART OF THE FIX ***
# Preprocess the RGB NumPy array (img_rgb_np) for MiDaS
input_batch = midas_transforms(img_rgb_np).to(device)

with torch.no_grad():
    prediction = model_midas(input_batch)

    # Resize depth map to original image size
    depth_map = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(img_h, img_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

# Note: MiDaS produces inverse depth. Higher values mean *closer* to the camera.

# --- 5. Task 3: Occlusion Reasoning ---
print("Performing occlusion reasoning...")
# Overlap Threshold
IOU_THRESHOLD = 0.2

# 5a: Calculate median depth for all detected vehicles
for det in detections:
    # Get bounding box coordinates as integers
    x1, y1, x2, y2 = det['bbox'].astype(int)

    # Ensure coordinates are within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

    # Slice the depth map to get the region inside the bounding box
    depth_patch = depth_map[y1:y2, x1:x2]

    # Calculate and store the median depth value
    if depth_patch.size > 0:
        det['median_depth'] = np.median(depth_patch)
    else:
        det['median_depth'] = 0  # Handle empty patches if any

# 5b: Compare overlapping boxes
for i in range(len(detections)):
    for j in range(i + 1, len(detections)):
        det_i = detections[i]
        det_j = detections[j]

        # Calculate IoU
        iou = calculate_iou(det_i['bbox'], det_j['bbox'])

        if iou > IOU_THRESHOLD:
            # The boxes overlap. Check which one is in front.
            # Remember: Higher depth value from MiDaS = CLOSER
            if det_i['median_depth'] > det_j['median_depth']:
                # i is in front, so j is occluded (by i)
                det_j['occluded'] = True
            else:
                # j is in front, so i is occluded (by j)
                det_i['occluded'] = True

# --- 6. Task 4 & Expected Result 1: Visualization ---
print("Generating visualization...")
vis_image = img_cv.copy()  # Draw on the original BGR image

# Colors (BGR format)
COLOR_GREEN_VISIBLE = (0, 255, 0)
COLOR_RED_OCCLUDED = (0, 0, 255)

for det in detections:
    x1, y1, x2, y2 = det['bbox'].astype(int)

    # Determine color based on occlusion status
    color = COLOR_RED_OCCLUDED if det['occluded'] else COLOR_GREEN_VISIBLE
    status = "Occluded" if det['occluded'] else "Visible"

    # Draw the bounding box
    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

    # Create label text
    label = f"{model_yolo.names[det['cls']]}"
    status_label = f"{status} (Depth: {det['median_depth']:.2f})"

    # Get text size to draw a background
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (w_s, h_s), _ = cv2.getTextSize(
        status_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    max_w = max(w, w_s)

    # Add text background
    cv2.rectangle(vis_image, (x1, y1 - 40), (x1 + max_w + 10, y1), color, -1)

    # Add text
    cv2.putText(vis_image, label, (x1 + 5, y1 - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(vis_image, status_label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# --- 7. Expected Result 2: Table ---
print("\n" + "="*50)
print("              Detection Results Table")
print("="*50)
print(f"{'ID':<5} | {'Class':<12} | {'Median Depth':<15} | {'Status':<10}")
print("-"*50)

for i, det in enumerate(detections):
    cls_name = model_yolo.names[det['cls']]
    status = "Occluded" if det['occluded'] else "Visible"
    print(
        f"{i:<5} | {cls_name:<12} | {det['median_depth']:<15.2f} | {status:<10}")
print("="*50)


# --- 8. Expected Result 3: Simple Conclusion ---
print("\n--- Conclusion ---")
print("Depth helps distinguish overlapping vehicles by providing a relative distance for each object.")
print("By comparing the median depth of overlapping bounding boxes (IoU > 0.2), we can infer occlusion.")
print("In this implementation (using MiDaS), a *higher* depth value means the object is *closer*,")
print("therefore the object with the lower median depth is marked as 'Occluded'.")


# --- 9. Display Results ---
print("\nDisplaying results. Close the window to exit.")
# Convert BGR (cv2) to RGB (matplotlib) for final display
vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 10))

# Subplot 1: Occlusion Reasoning Visualization
plt.subplot(1, 2, 1)
plt.title("Occlusion Reasoning (Green=Visible, Red=Occluded)")
plt.imshow(vis_image_rgb)
plt.axis('off')

# Subplot 2: Depth Map
plt.subplot(1, 2, 2)
plt.title("Generated Depth Map (Lighter = Closer)")
plt.imshow(depth_map, cmap='plasma')
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nTask complete.")
