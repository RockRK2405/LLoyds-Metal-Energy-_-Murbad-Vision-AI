import cv2
import numpy as np
import os
import sys
import argparse
from ultralytics import YOLO

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[INFO] tqdm not installed. Using basic progress. Install with: pip3 install tqdm")
    sys.stdout.flush()

# ──────────────── CLI ARGUMENTS ────────────────
parser = argparse.ArgumentParser(description="Auto-label images for PPE detection")
parser.add_argument("--image_dir", type=str, default=None,
                    help="Path to the image directory (absolute or relative to script dir)")
parser.add_argument("--label_dir", type=str, default=None,
                    help="Path to the label output directory (absolute or relative to script dir)")
args = parser.parse_args()

# ──────────────── CONFIGURATION ────────────────
# Resolve paths relative to the project root (parent of PPE_Project)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # "Lloyds Folder"

# YOLO model - sits in the project root
MODEL_PATH = os.path.join(PROJECT_ROOT, "yolo11x.pt")

# Image and label directories (use CLI args if provided, else defaults)
if args.image_dir:
    IMAGE_DIR = os.path.join(SCRIPT_DIR, args.image_dir) if not os.path.isabs(args.image_dir) else args.image_dir
else:
    IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")

if args.label_dir:
    LABEL_DIR = os.path.join(SCRIPT_DIR, args.label_dir) if not os.path.isabs(args.label_dir) else args.label_dir
else:
    LABEL_DIR = os.path.join(SCRIPT_DIR, "labels")

NAMES_FILE = os.path.join(SCRIPT_DIR, "lewl.names")

# YOLO detection settings
CONFIDENCE_THRESHOLD = 0.25   # Minimum confidence for person detections
IOU_THRESHOLD = 0.45          # NMS IoU threshold

# ──────────────── CLASS MAPPING ────────────────
# Read classes from lewl.names:  line 0 = "p" (PPE), line 1 = "n" (no PPE)
with open(NAMES_FILE, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
print(f"Classes from lewl.names: {CLASS_NAMES}"); sys.stdout.flush()
# class_id 0 → "p" (human WITH yellow helmet / PPE)
# class_id 1 → "n" (human WITHOUT yellow helmet / PPE)

# ──────────────── HELMET COLOR DETECTION (YELLOW + RED) ────────────────
# --- YELLOW helmet ranges ---
# Yellow helmets can appear from golden-yellow to bright yellow
LOWER_YELLOW_1 = np.array([15, 80, 80])    # Lower bound (orangish-yellow)
UPPER_YELLOW_1 = np.array([35, 255, 255])   # Upper bound (yellow-green)
# Some helmets under bright/fluorescent lighting can shift toward greenish-yellow
LOWER_YELLOW_2 = np.array([35, 60, 80])
UPPER_YELLOW_2 = np.array([45, 255, 255])

# --- RED helmet ranges ---
# Red wraps around hue 0/180 in HSV, so we need TWO ranges
LOWER_RED_1 = np.array([0, 70, 50])         # Low-hue reds (0-10)
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 70, 50])       # High-hue reds (170-180)
UPPER_RED_2 = np.array([180, 255, 255])

# Head region: proportion of bounding box height to check for helmet
# We look at the top 30% of the person bounding box (head + shoulders area)
HEAD_REGION_RATIO = 0.30

# Minimum helmet pixel percentage in the HEAD REGION to classify as PPE
# Since we're only looking at the head region, even 1-2% is meaningful
HELMET_THRESHOLD_PERCENT = 1.5

# Minimum absolute helmet pixel count (prevents noise triggering on tiny ROIs)
MIN_HELMET_PIXEL_COUNT = 8

# ──────────────── HELPERS ────────────────
def get_box_overlap(boxA, boxB):
    """
    Calculate the overlap ratio of boxA with boxB.
    Defined as: Area(Intersection) / Area(boxA).
    """
    ix1 = max(boxA[0], boxB[0])
    iy1 = max(boxA[1], boxB[1])
    ix2 = min(boxA[2], boxB[2])
    iy2 = min(boxA[3], boxB[3])
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
        
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    if boxA_area == 0:
        return 0.0
    return inter_area / float(boxA_area)

# ──────────────── SETUP ────────────────
os.makedirs(LABEL_DIR, exist_ok=True)

# Load YOLO11x model (accurate person detector)
print(f"Loading person detection model from: {MODEL_PATH}"); sys.stdout.flush()
model = YOLO(MODEL_PATH)

# Load fine-tuned YOLO11n PPE model (for hardhat detections)
BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")
print(f"Loading fine-tuned PPE model from: {BEST_MODEL_PATH}"); sys.stdout.flush()
best_model = YOLO(BEST_MODEL_PATH)
print("Models loaded successfully!"); sys.stdout.flush()

# ──────────────── PROCESS IMAGES ────────────────
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
])
total_images = len(image_files)
print(f"Found {total_images} images to process."); sys.stdout.flush()

# Statistics tracking
stats = {"total_detections": 0, "class_p": 0, "class_n": 0, "images_processed": 0, "images_skipped": 0}

# Wrap with tqdm progress bar if available
iterator = enumerate(image_files)
if HAS_TQDM:
    pbar = tqdm(iterator, total=total_images, desc="Auto-labeling", unit="img",
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}")
else:
    pbar = iterator

for idx, image_file in pbar:
    image_path = os.path.join(IMAGE_DIR, image_file)
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [SKIP] Could not read: {image_path}")
        stats["images_skipped"] += 1
        continue

    img_height, img_width = img.shape[:2]
    label_filename = os.path.splitext(image_file)[0] + ".txt"
    label_filepath = os.path.join(LABEL_DIR, label_filename)

    # Run YOLO person detection (COCO class 0 = person)
    results = model.predict(
        source=img,
        classes=[0],          # Only detect persons
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False
    )

    # Detect hardhats (0) and no-hardhats (2) using best_model
    ppe_results = best_model.predict(
        source=img,
        classes=[0, 2],       # 0 = Hardhat, 2 = NO-Hardhat
        conf=0.15,            # Lower threshold for smaller objects
        iou=0.45,
        verbose=False
    )

    # Extract detected hardhats and no-hardhats
    detected_hardhats = []
    detected_no_hardhats = []
    for ppe_result in ppe_results:
        for b in ppe_result.boxes:
            cls = int(b.cls[0])
            box_coords = list(map(int, b.xyxy[0]))
            if cls == 0:
                detected_hardhats.append(box_coords)
            elif cls == 2:
                detected_no_hardhats.append(box_coords)

    annotations = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates (pixels)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)

            box_h = y2 - y1
            box_w = x2 - x1

            # Skip extremely small detections (likely noise)
            if box_h < 5 or box_w < 3:
                continue

            # ── HELMET DETECTION: Check best_model detections first ──
            matched_hardhat = False
            matched_no_hardhat = False
            person_top_boundary = y1 + int(box_h * 0.40)

            # Check if any detected hardhat box belongs to this person
            for h_box in detected_hardhats:
                overlap = get_box_overlap(h_box, [x1, y1, x2, y2])
                if overlap > 0.5:
                    h_center_y = (h_box[1] + h_box[3]) / 2.0
                    if h_center_y <= person_top_boundary:
                        matched_hardhat = True
                        break

            # Check if any detected NO-hardhat box belongs to this person
            for nh_box in detected_no_hardhats:
                overlap = get_box_overlap(nh_box, [x1, y1, x2, y2])
                if overlap > 0.5:
                    nh_center_y = (nh_box[1] + nh_box[3]) / 2.0
                    if nh_center_y <= person_top_boundary:
                        matched_no_hardhat = True
                        break

            has_helmet = False

            if matched_hardhat:
                has_helmet = True
            elif matched_no_hardhat:
                has_helmet = False
            else:
                # ── FALLBACK: HSV-based color detection (yellow + red) ──
                # Extract the top portion of the bounding box (head + shoulders)
                head_y2 = y1 + int(box_h * HEAD_REGION_RATIO)
                head_y2 = min(head_y2, y2)  # Don't exceed the box
                head_roi = img[y1:head_y2, x1:x2]

                if head_roi.size > 0:
                    # Convert head region to HSV
                    hsv_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)

                    # Create YELLOW mask (two ranges)
                    y_mask1 = cv2.inRange(hsv_head, LOWER_YELLOW_1, UPPER_YELLOW_1)
                    y_mask2 = cv2.inRange(hsv_head, LOWER_YELLOW_2, UPPER_YELLOW_2)
                    yellow_mask = cv2.bitwise_or(y_mask1, y_mask2)

                    # Create RED mask (two ranges)
                    r_mask1 = cv2.inRange(hsv_head, LOWER_RED_1, UPPER_RED_1)
                    r_mask2 = cv2.inRange(hsv_head, LOWER_RED_2, UPPER_RED_2)
                    red_mask = cv2.bitwise_or(r_mask1, r_mask2)

                    # Combine: ANY helmet color (yellow OR red) = PPE
                    helmet_mask = cv2.bitwise_or(yellow_mask, red_mask)

                    # Apply morphological operations to reduce noise
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    helmet_mask = cv2.morphologyEx(helmet_mask, cv2.MORPH_OPEN, kernel)

                    # Count helmet-colored pixels
                    helmet_pixel_count = np.sum(helmet_mask > 0)
                    total_head_pixels = head_roi.shape[0] * head_roi.shape[1]

                    if total_head_pixels > 0:
                        helmet_percentage = (helmet_pixel_count / total_head_pixels) * 100

                        # Classify: need BOTH percentage AND absolute count thresholds
                        if helmet_percentage >= HELMET_THRESHOLD_PERCENT and helmet_pixel_count >= MIN_HELMET_PIXEL_COUNT:
                            has_helmet = True

            # Assign class
            if has_helmet:
                class_id = 0  # "p" — human with PPE
            else:
                class_id = 1  # "n" — human without PPE

            # Convert bounding box to YOLO format (normalized center x, center y, width, height)
            x_center = (x1 + x2) / (2.0 * img_width)
            y_center = (y1 + y2) / (2.0 * img_height)
            norm_width = (x2 - x1) / float(img_width)
            norm_height = (y2 - y1) / float(img_height)

            annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

            # Update stats
            stats["total_detections"] += 1
            if class_id == 0:
                stats["class_p"] += 1
            else:
                stats["class_n"] += 1

    # Write label file (even if empty — empty file means no humans detected)
    with open(label_filepath, 'w') as f:
        if annotations:
            f.write("\n".join(annotations) + "\n")

    stats["images_processed"] += 1

    # Update progress bar description with live stats
    if HAS_TQDM:
        pbar.set_postfix({
            "det": stats['total_detections'],
            "p": stats['class_p'],
            "n": stats['class_n']
        })
    else:
        # Fallback: print progress every 50 images
        if (idx + 1) % 50 == 0 or (idx + 1) == total_images:
            pct = (idx + 1) / total_images * 100
            bar_len = 30
            filled = int(bar_len * (idx + 1) // total_images)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r  [{bar}] {pct:5.1f}% [{idx+1}/{total_images}] "
                  f"det={stats['total_detections']} p={stats['class_p']} n={stats['class_n']}",
                  end='', flush=True)
            if (idx + 1) == total_images:
                print()  # newline at end

# ──────────────── SUMMARY ────────────────
print("\n" + "=" * 60)
print("AUTO-LABELING COMPLETE")
print("=" * 60)
print(f"  Images processed : {stats['images_processed']}")
print(f"  Images skipped   : {stats['images_skipped']}")
print(f"  Total detections : {stats['total_detections']}")
print(f"  Class 'p' (PPE)  : {stats['class_p']}")
print(f"  Class 'n' (no PPE): {stats['class_n']}")
print(f"  Labels saved to  : {LABEL_DIR}")
print(f"  Names file used  : {NAMES_FILE}")
print("=" * 60)
