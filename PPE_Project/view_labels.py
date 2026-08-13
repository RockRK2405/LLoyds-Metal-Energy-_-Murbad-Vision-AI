import cv2
import os
import sys
import random

# ──────────────── CONFIGURATION ────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")
LABEL_DIR = os.path.join(SCRIPT_DIR, "labels")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "preview")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class names and colors (BGR)
CLASS_NAMES = {0: "p (PPE)", 1: "n (No PPE)"}
CLASS_COLORS = {
    0: (0, 255, 0),    # Green for PPE
    1: (0, 0, 255),    # Red for No PPE
}

# How many sample images to visualize
NUM_SAMPLES = 10

# ──────────────── FIND IMAGES WITH DETECTIONS ────────────────
label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".txt")]

# Pick labels that have at least one detection (non-empty files)
non_empty_labels = []
for lf in label_files:
    filepath = os.path.join(LABEL_DIR, lf)
    if os.path.getsize(filepath) > 0:
        non_empty_labels.append(lf)

# Randomly sample
random.seed(42)
samples = random.sample(non_empty_labels, min(NUM_SAMPLES, len(non_empty_labels)))

print(f"Generating {len(samples)} preview images with bounding boxes...")
print(f"Output: {OUTPUT_DIR}/\n")

for label_file in samples:
    base_name = os.path.splitext(label_file)[0]

    # Find matching image
    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = os.path.join(IMAGE_DIR, base_name + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        print(f"  [SKIP] No image found for {label_file}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"  [SKIP] Could not read {img_path}")
        continue

    img_h, img_w = img.shape[:2]

    # Read label file
    label_path = os.path.join(LABEL_DIR, label_file)
    with open(label_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    p_count = 0
    n_count = 0

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Convert YOLO format back to pixel coordinates
        x1 = int((x_center - width / 2) * img_w)
        y1 = int((y_center - height / 2) * img_h)
        x2 = int((x_center + width / 2) * img_w)
        y2 = int((y_center + height / 2) * img_h)

        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        label_text = CLASS_NAMES.get(class_id, f"cls_{class_id}")

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label_text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if class_id == 0:
            p_count += 1
        else:
            n_count += 1

    # Save preview
    out_path = os.path.join(OUTPUT_DIR, f"preview_{base_name}.jpg")
    cv2.imwrite(out_path, img)
    print(f"  ✓ {base_name} → {p_count} PPE (green), {n_count} no-PPE (red)")

print(f"\nDone! Preview images saved to: {OUTPUT_DIR}/")
print("Open the 'preview' folder to view them.")
