import cv2
import os
import json

# Configuration
IMAGE_DIR = "PPE_Project/images"
LABEL_DIR = "PPE_Project/labels"
CLASSES = ["p", "n"] # p: PPE/helmet, n: No PPE/helmet

# Ensure label directory exists
os.makedirs(LABEL_DIR, exist_ok=True)

# Load existing annotations if they exist
existing_annotations = {}
for filename in os.listdir(LABEL_DIR):
    if filename.endswith(".txt"):
        img_name = filename[:-4] # Remove .txt extension
        filepath = os.path.join(LABEL_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                annotations = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        bbox = [float(p) for p in parts[1:]]
                        annotations.append({"class_id": class_id, "bbox": bbox})
                existing_annotations[img_name] = annotations
        except Exception as e:
            print(f"Error loading annotation for {filename}: {e}")

# Get list of images
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))])

current_image_index = 0
drawing = False
start_point = None
end_point = None
current_boxes = []
current_image_name = ""

def get_yolo_format(box, img_width, img_height, class_id):
    x_center = (box[0] + box[2]) / 2 / img_width
    y_center = (box[1] + box[3]) / 2 / img_height
    width = (box[2] - box[0]) / img_width
    height = (box[3] - box[1]) / img_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def draw_boxes(img, boxes):
    img_with_boxes = img.copy()
    for box_data in boxes:
        class_id = box_data["class_id"]
        bbox = box_data["bbox"]
        x_min, y_min, x_max, y_max = bbox

        # Convert normalized YOLO format back to pixel coordinates for drawing
        img_h, img_w = img.shape[:2]
        x_min_px = int(x_min * img_w)
        y_min_px = int(y_min * img_h)
        x_max_px = int(x_max * img_w)
        y_max_px = int(y_max * img_h)

        color = (0, 255, 0) if class_id == 0 else (0, 0, 255) # Green for 'p', Red for 'n'
        cv2.rectangle(img_with_boxes, (x_min_px, y_min_px), (x_max_px, y_max_px), color, 2)
        cv2.putText(img_with_boxes, CLASSES[class_id], (x_min_px, y_min_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img_with_boxes

def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, drawing, current_boxes, current_image_name, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        if start_point and end_point and start_point != end_point:
            # Ensure box coordinates are valid
            x_min, y_min = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
            x_max, y_max = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])

            # Prompt for class
            print("\nClasses: 0 for 'p' (PPE/helmet), 1 for 'n' (No PPE/helmet)")
            class_input = input("Enter class (0 or 1): ")
            try:
                class_id = int(class_input)
                if class_id not in [0, 1]:
                    print("Invalid class ID. Please enter 0 or 1.")
                    return
                current_boxes.append({"class_id": class_id, "bbox": [x_min, y_min, x_max, y_max]})
            except ValueError:
                print("Invalid input. Please enter a number.")
        start_point = None
        end_point = None

def save_annotations():
    global current_boxes, current_image_name
    if not current_image_name:
        return

    img_h, img_w = img.shape[:2]
    yolo_annotations = []
    for box_data in current_boxes:
        yolo_annotations.append(get_yolo_format(box_data["bbox"], img_w, img_h, box_data["class_id"]))

    annotation_filename = os.path.join(LABEL_DIR, f"{current_image_name}.txt")
    with open(annotation_filename, 'w') as f:
        f.write("\n".join(yolo_annotations))
    print(f"Annotations saved for {current_image_name}.txt")

# Main loop
while True:
    if current_image_index >= len(image_files):
        print("All images processed. Exiting.")
        break

    current_image_name = image_files[current_image_index][:-4] # Remove extension
    image_path = os.path.join(IMAGE_DIR, image_files[current_image_index])
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image {image_path}. Skipping.")
        current_image_index += 1
        continue

    # Load existing boxes if available
    if current_image_name in existing_annotations:
        current_boxes = existing_annotations[current_image_name]
    else:
        current_boxes = []

    # Draw existing boxes
    img_display = draw_boxes(img, current_boxes)

    cv2.namedWindow("Annotation Tool")
    cv2.setMouseCallback("Annotation Tool", mouse_callback)

    while True:
        img_display_current = draw_boxes(img, current_boxes)
        if drawing and start_point:
            cv2.rectangle(img_display_current, start_point, end_point, (255, 0, 0), 2) # Blue for drawing

        cv2.imshow("Annotation Tool", img_display_current)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('d'): # Next image
            save_annotations()
            current_image_index += 1
            break
        elif key == ord('a'): # Previous image
            save_annotations()
            current_image_index -= 1
            if current_image_index < 0:
                current_image_index = 0
            break
        elif key == ord('s'): # Save current annotations
            save_annotations()
        elif key == ord('q'): # Quit
            save_annotations()
            cv2.destroyAllWindows()
            exit()
        elif key == ord('r'): # Reset boxes for current image
            current_boxes = []
            print("Boxes reset for current image.")
        elif key == 27: # ESC key to cancel drawing
            drawing = False
            start_point = None
            end_point = None
            print("Drawing cancelled.")

    cv2.destroyAllWindows()

cv2.destroyAllWindows()
print("Annotation process finished.")