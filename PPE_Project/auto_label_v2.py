#!/usr/bin/env python3
"""
auto_label_v2.py — Redesigned PPE Detection Pipeline
=====================================================
Multi-stage helmet verification for industrial safety helmet classification.

Pipeline Architecture:
  1. Person detection via YOLO11x (COCO class 0)
  2. Adaptive head-region extraction (pose-aware)
  3. Tier 1: Fine-tuned PPE model (best.pt) for hardhat/no-hardhat
  4. Tier 2: Multi-color HSV analysis (yellow + white + red) with:
     - Contour-based shape validation (circularity, aspect ratio)
     - Spatial concentration checks
     - Morphological noise reduction
  5. Tier 3: Confidence score fusion from multiple cues
  6. Adaptive thresholds for distant/small workers

Classes:
  0 → Person WITH PPE helmet (yellow / white / red hard hat)
  1 → Person WITHOUT PPE helmet

Output:
  labels_v2/ directory (existing labels/ untouched)
"""

import cv2
import numpy as np
import os
import sys
import argparse
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from ultralytics import YOLO

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("[INFO] tqdm not installed. Using basic progress. Install with: pip3 install tqdm")
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HelmetCandidate:
    """Represents a potential helmet detection from contour analysis."""
    contour: np.ndarray
    area: float
    circularity: float
    aspect_ratio: float
    centroid: Tuple[int, int]
    color_label: str  # "yellow", "white", or "red"
    pixel_count: int
    bounding_rect: Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class HelmetVerdict:
    """Final helmet classification result with confidence breakdown."""
    has_helmet: bool
    confidence: float
    method: str  # "ppe_model", "color_analysis", "default"
    color_detected: str = ""
    details: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """
    Centralized configuration for the PPE detection pipeline.
    All tunable parameters live here for easy adjustment.
    """

    # ── Detection thresholds ──
    PERSON_CONF_THRESHOLD = 0.25     # YOLO person detection confidence
    PERSON_IOU_THRESHOLD = 0.45      # NMS IoU threshold for persons
    PPE_MODEL_CONF = 0.15            # Fine-tuned PPE model confidence
    PPE_MODEL_IOU = 0.45             # PPE model NMS IoU

    # ── Adaptive head region ──
    HEAD_RATIO_STANDING = 0.20       # Top 20% for upright persons (h/w > 1.8)
    HEAD_RATIO_MODERATE = 0.25       # Top 25% for moderate aspect ratio
    HEAD_RATIO_CROUCHING = 0.30      # Top 30% for crouching/sitting (h/w < 1.2)
    HEAD_LATERAL_PAD = 0.05          # 5% lateral expansion of head crop

    # ── Minimum person size filters ──
    MIN_PERSON_HEIGHT_PX = 15        # Skip detections smaller than this
    MIN_PERSON_WIDTH_PX = 8          # Skip detections narrower than this
    TINY_PERSON_THRESHOLD = 30       # Below this height: PPE model only, no HSV
    SMALL_PERSON_THRESHOLD = 80      # Below this height: relaxed thresholds

    # ── HSV color ranges for helmet detection ──
    # YELLOW: Tightened to exclude golden/amber machinery tones
    YELLOW_RANGES = [
        (np.array([18, 100, 100]), np.array([32, 255, 255])),   # Core yellow
        (np.array([32,  80, 100]), np.array([42, 255, 255])),   # Yellow-green shift
    ]

    # WHITE: Low saturation, high value — needs shape validation
    WHITE_RANGES = [
        (np.array([0,   0, 180]), np.array([180, 50, 255])),    # Pure white
        (np.array([0,   0, 160]), np.array([180, 60, 240])),    # Off-white / matte
    ]

    # RED: Two hue bands (wraps around 0/180), saturation floor raised to 100
    RED_RANGES = [
        (np.array([0,  100, 70]),  np.array([8,  255, 255])),   # Low-hue red
        (np.array([172, 100, 70]), np.array([180, 255, 255])),  # High-hue red
    ]

    # ── Contour shape validation ──
    MIN_CIRCULARITY = 0.20           # Helmets are roundish (relaxed for partial views)
    MIN_CIRCULARITY_SMALL = 0.12     # Even more relaxed for small workers
    MIN_ASPECT_RATIO = 0.4           # Minimum w/h ratio (rejects thin vertical strips)
    MAX_ASPECT_RATIO = 3.0           # Maximum w/h ratio (rejects thin horizontal strips)
    MIN_CONTOUR_AREA_FRACTION = 0.02 # Min contour area as fraction of head region
    MIN_CONTOUR_AREA_FRACTION_SMALL = 0.01  # Relaxed for small workers

    # ── Spatial concentration ──
    MIN_CONCENTRATION_RATIO = 0.15   # Helmet blob must fill at least 15% of its bounding rect
    CENTROID_UPPER_FRACTION = 0.75   # Contour centroid must be in upper 75% of head region

    # ── Confidence fusion weights ──
    WEIGHT_COLOR_PERCENTAGE = 0.25
    WEIGHT_CIRCULARITY = 0.25
    WEIGHT_CONCENTRATION = 0.20
    WEIGHT_SIZE_MATCH = 0.15
    WEIGHT_POSITION = 0.15

    # ── Final classification thresholds ──
    HELMET_CONF_THRESHOLD = 0.35         # Minimum fused confidence for helmet
    HELMET_CONF_THRESHOLD_SMALL = 0.25   # Relaxed for small workers
    WHITE_HELMET_CONF_BONUS = -0.05      # White is noisier, slightly penalize

    # ── Morphological kernel sizes ──
    MORPH_KERNEL_SMALL = (3, 3)
    MORPH_KERNEL_MEDIUM = (5, 5)

    # ── PPE model overlap matching ──
    PPE_OVERLAP_THRESHOLD = 0.4          # Minimum overlap for PPE box → person matching
    PPE_HEAD_BOUNDARY_RATIO = 0.45       # PPE box center must be in top 45% of person


# ══════════════════════════════════════════════════════════════════════════════
# HEAD REGION EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_head_region(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    img_h: int, img_w: int
) -> Optional[np.ndarray]:
    """
    Extract the head region from a person bounding box using adaptive sizing.

    The head ratio is chosen based on the person's aspect ratio:
      - Tall/standing persons (h/w > 1.8): use top 20%
      - Moderate aspect ratio: use top 25%
      - Crouching/sitting (h/w < 1.2): use top 30%

    Lateral padding is added to catch helmets at bounding box edges.

    Returns:
        Head region crop as a numpy array, or None if too small.
    """
    box_h = y2 - y1
    box_w = x2 - x1

    if box_h < 5 or box_w < 3:
        return None

    # Adaptive head ratio based on person pose
    aspect = box_h / max(box_w, 1)
    if aspect > 1.8:
        head_ratio = Config.HEAD_RATIO_STANDING
    elif aspect > 1.2:
        head_ratio = Config.HEAD_RATIO_MODERATE
    else:
        head_ratio = Config.HEAD_RATIO_CROUCHING

    # Calculate head region boundaries
    head_y2 = y1 + int(box_h * head_ratio)
    head_y2 = min(head_y2, y2)

    # Add lateral padding to catch edge helmets
    pad_x = int(box_w * Config.HEAD_LATERAL_PAD)
    head_x1 = max(0, x1 - pad_x)
    head_x2 = min(img_w, x2 + pad_x)

    # Clamp to image boundaries
    head_y1 = max(0, y1)
    head_y2 = min(img_h, head_y2)

    head_roi = img[head_y1:head_y2, head_x1:head_x2]

    if head_roi.size == 0:
        return None

    return head_roi


# ══════════════════════════════════════════════════════════════════════════════
# HSV COLOR SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def create_color_mask(
    hsv_img: np.ndarray,
    color_ranges: List[Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """
    Create a binary mask for a given set of HSV color ranges.

    Combines multiple ranges with bitwise OR to handle colors that span
    multiple HSV regions (e.g., red wraps around hue 0/180).
    """
    combined_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv_img, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask


def apply_morphological_cleanup(
    mask: np.ndarray,
    is_small_person: bool = False
) -> np.ndarray:
    """
    Apply morphological operations to reduce noise in color masks.

    For normal-sized persons:
      1. Morphological OPEN to remove salt noise (small false positives)
      2. Morphological CLOSE to fill small gaps in helmet regions

    For small persons:
      1. Gentler open with smaller kernel
      2. Dilate to recover small helmet regions that might be eroded
    """
    if is_small_person:
        kernel_small = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, Config.MORPH_KERNEL_SMALL
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        mask = cv2.dilate(mask, kernel_small, iterations=1)
    else:
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, Config.MORPH_KERNEL_SMALL
        )
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, Config.MORPH_KERNEL_MEDIUM
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    return mask


# ══════════════════════════════════════════════════════════════════════════════
# CONTOUR-BASED SHAPE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_contours(
    mask: np.ndarray,
    color_label: str,
    head_h: int,
    head_w: int,
    is_small_person: bool = False
) -> List[HelmetCandidate]:
    """
    Find contours in the color mask and validate each one against helmet-like
    shape criteria:

      1. Circularity: 4π × area / perimeter² — helmets are roundish
      2. Aspect ratio: width / height — rejects thin pipes and strips
      3. Area: Must be a meaningful fraction of the head region
      4. Position: Centroid must be in the upper portion of the head region

    Returns a list of HelmetCandidate objects that pass all filters.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    head_area = max(head_h * head_w, 1)
    min_circ = Config.MIN_CIRCULARITY_SMALL if is_small_person else Config.MIN_CIRCULARITY
    min_area_frac = (Config.MIN_CONTOUR_AREA_FRACTION_SMALL
                     if is_small_person else Config.MIN_CONTOUR_AREA_FRACTION)

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # ── Area filter ──
        if area < head_area * min_area_frac:
            continue

        # ── Circularity ──
        if perimeter > 0:
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        else:
            continue

        if circularity < min_circ:
            continue

        # ── Bounding rect aspect ratio ──
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bh == 0:
            continue
        aspect_ratio = bw / float(bh)

        if aspect_ratio < Config.MIN_ASPECT_RATIO or aspect_ratio > Config.MAX_ASPECT_RATIO:
            continue

        # ── Centroid position: must be in upper portion of head region ──
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if cy > head_h * Config.CENTROID_UPPER_FRACTION:
            continue

        # ── Pixel count in this contour ──
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        pixel_count = int(np.sum((mask > 0) & (contour_mask > 0)))

        candidates.append(HelmetCandidate(
            contour=contour,
            area=area,
            circularity=circularity,
            aspect_ratio=aspect_ratio,
            centroid=(cx, cy),
            color_label=color_label,
            pixel_count=pixel_count,
            bounding_rect=(bx, by, bw, bh),
        ))

    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL CONCENTRATION CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_spatial_concentration(candidate: HelmetCandidate) -> float:
    """
    Verify that the helmet-colored pixels form a compact cluster, not
    scattered noise.

    Computes: contour_area / bounding_rect_area
    A real helmet fills a significant portion of its bounding rectangle.
    Scattered noise will have a very low fill ratio.

    Returns:
        Concentration ratio (0.0 to 1.0). Higher = more compact.
    """
    bx, by, bw, bh = candidate.bounding_rect
    rect_area = max(bw * bh, 1)
    concentration = candidate.area / rect_area
    return concentration


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCORE FUSION
# ══════════════════════════════════════════════════════════════════════════════

def compute_helmet_confidence(
    candidate: HelmetCandidate,
    head_h: int,
    head_w: int,
    total_head_pixels: int,
    is_small_person: bool = False
) -> float:
    """
    Fuse multiple cues into a single confidence score for a helmet candidate.

    Cues and their weights:
      1. Color pixel percentage — what fraction of the head region is helmet-colored
      2. Circularity — how round the contour is (helmets are round)
      3. Spatial concentration — how compact the colored region is
      4. Size match — how well the contour area matches expected helmet size
      5. Position — how centered and high the contour is in the head region

    Returns:
        Fused confidence score (0.0 to 1.0).
    """
    head_area = max(total_head_pixels, 1)

    # 1. Color percentage score (clamped, normalized)
    color_pct = (candidate.pixel_count / head_area) * 100
    # Helmet typically occupies 3-40% of head region
    color_score = min(color_pct / 15.0, 1.0)  # Normalize to 0-1, saturates at 15%

    # 2. Circularity score (already 0-1 range, but cap it)
    circ_score = min(candidate.circularity / 0.7, 1.0)  # Perfect circle = 1.0

    # 3. Spatial concentration
    concentration = check_spatial_concentration(candidate)
    conc_score = min(concentration / 0.5, 1.0)  # Good fill = 0.5+

    # 4. Size match — expected helmet is roughly 30-70% of head width
    bx, by, bw, bh = candidate.bounding_rect
    expected_w = head_w * 0.5
    size_ratio = min(bw, expected_w) / max(bw, expected_w, 1)
    size_score = size_ratio

    # 5. Position score — helmet should be centered horizontally, high vertically
    cx, cy = candidate.centroid
    # Horizontal: penalty for being far from center
    h_center = head_w / 2.0
    h_dist = abs(cx - h_center) / max(head_w, 1)
    h_score = max(0, 1.0 - h_dist * 2)

    # Vertical: reward for being in the upper region
    v_score = max(0, 1.0 - (cy / max(head_h, 1)))

    position_score = (h_score + v_score) / 2.0

    # ── Weighted fusion ──
    confidence = (
        Config.WEIGHT_COLOR_PERCENTAGE * color_score +
        Config.WEIGHT_CIRCULARITY * circ_score +
        Config.WEIGHT_CONCENTRATION * conc_score +
        Config.WEIGHT_SIZE_MATCH * size_score +
        Config.WEIGHT_POSITION * position_score
    )

    # Apply white helmet penalty (white is noisier — more false positives)
    if candidate.color_label == "white":
        confidence += Config.WHITE_HELMET_CONF_BONUS

    return max(0.0, min(1.0, confidence))


# ══════════════════════════════════════════════════════════════════════════════
# PPE MODEL MATCHING
# ══════════════════════════════════════════════════════════════════════════════

def get_box_overlap(boxA: List[int], boxB: List[int]) -> float:
    """
    Calculate the overlap ratio of boxA with boxB.
    Defined as: Area(Intersection) / Area(boxA).

    This is NOT IoU — it measures what fraction of boxA is covered by boxB.
    We use this because a small hardhat box should have high overlap with
    the larger person box, even if the person box is much bigger.
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


def match_ppe_detections(
    person_box: Tuple[int, int, int, int],
    detected_hardhats: List[List[int]],
    detected_no_hardhats: List[List[int]]
) -> Tuple[bool, bool]:
    """
    Match PPE model detections (hardhats / no-hardhats) to a specific person.

    A PPE detection is matched to a person if:
      1. Its overlap with the person box exceeds PPE_OVERLAP_THRESHOLD
      2. Its vertical center is in the upper PPE_HEAD_BOUNDARY_RATIO of the person

    Returns:
        (matched_hardhat: bool, matched_no_hardhat: bool)
    """
    x1, y1, x2, y2 = person_box
    box_h = y2 - y1
    head_boundary = y1 + int(box_h * Config.PPE_HEAD_BOUNDARY_RATIO)

    matched_hardhat = False
    matched_no_hardhat = False

    for h_box in detected_hardhats:
        overlap = get_box_overlap(h_box, [x1, y1, x2, y2])
        if overlap > Config.PPE_OVERLAP_THRESHOLD:
            h_center_y = (h_box[1] + h_box[3]) / 2.0
            if h_center_y <= head_boundary:
                matched_hardhat = True
                break

    for nh_box in detected_no_hardhats:
        overlap = get_box_overlap(nh_box, [x1, y1, x2, y2])
        if overlap > Config.PPE_OVERLAP_THRESHOLD:
            nh_center_y = (nh_box[1] + nh_box[3]) / 2.0
            if nh_center_y <= head_boundary:
                matched_no_hardhat = True
                break

    return matched_hardhat, matched_no_hardhat


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-CUE HELMET ANALYSIS (TIER 2 + TIER 3)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_helmet_colors(
    head_roi: np.ndarray,
    is_small_person: bool = False
) -> HelmetVerdict:
    """
    Perform multi-color HSV analysis on the head region to detect helmets.

    This is the Tier 2 + Tier 3 analysis used when the PPE model (Tier 1)
    doesn't produce a match.

    Steps for each color (yellow, white, red):
      1. Create HSV mask
      2. Apply morphological cleanup
      3. Validate contours (shape, size, position)
      4. Check spatial concentration
      5. Compute confidence score

    The best candidate across all colors wins.

    Returns:
        HelmetVerdict with the final classification.
    """
    head_h, head_w = head_roi.shape[:2]
    total_head_pixels = head_h * head_w

    if total_head_pixels == 0:
        return HelmetVerdict(
            has_helmet=False, confidence=0.0, method="color_analysis",
            details={"reason": "empty_head_region"}
        )

    # Convert to HSV once
    hsv_head = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)

    # Analyze each color channel
    color_configs = [
        ("yellow", Config.YELLOW_RANGES),
        ("red",    Config.RED_RANGES),
        ("white",  Config.WHITE_RANGES),
    ]

    best_candidate = None
    best_confidence = 0.0
    all_details = {}

    for color_label, color_ranges in color_configs:
        # Step 1: Create color mask
        mask = create_color_mask(hsv_head, color_ranges)

        # Step 2: Morphological cleanup
        mask = apply_morphological_cleanup(mask, is_small_person)

        # Step 3: Validate contours
        candidates = validate_contours(
            mask, color_label, head_h, head_w, is_small_person
        )

        # Step 4 + 5: Score each candidate
        for candidate in candidates:
            confidence = compute_helmet_confidence(
                candidate, head_h, head_w, total_head_pixels, is_small_person
            )

            all_details[f"{color_label}_best_conf"] = max(
                all_details.get(f"{color_label}_best_conf", 0), confidence
            )

            if confidence > best_confidence:
                best_confidence = confidence
                best_candidate = candidate

    # ── Determine threshold ──
    threshold = (Config.HELMET_CONF_THRESHOLD_SMALL
                 if is_small_person else Config.HELMET_CONF_THRESHOLD)

    if best_candidate and best_confidence >= threshold:
        return HelmetVerdict(
            has_helmet=True,
            confidence=best_confidence,
            method="color_analysis",
            color_detected=best_candidate.color_label,
            details={
                "circularity": best_candidate.circularity,
                "pixel_count": best_candidate.pixel_count,
                "concentration": check_spatial_concentration(best_candidate),
                **all_details,
            }
        )
    else:
        return HelmetVerdict(
            has_helmet=False,
            confidence=best_confidence,
            method="color_analysis",
            details={
                "best_score": best_confidence,
                "threshold": threshold,
                **all_details,
            }
        )


# ══════════════════════════════════════════════════════════════════════════════
# FULL DETECTION PIPELINE (PER PERSON)
# ══════════════════════════════════════════════════════════════════════════════

def classify_person(
    img: np.ndarray,
    person_box: Tuple[int, int, int, int],
    detected_hardhats: List[List[int]],
    detected_no_hardhats: List[List[int]],
    img_h: int,
    img_w: int
) -> HelmetVerdict:
    """
    Full three-tier classification for a single detected person.

    Tier 1: PPE model match (highest confidence — trained detector)
    Tier 2: Multi-cue color analysis (HSV + shape + spatial)
    Tier 3: Confidence fusion determines final class

    For very small persons (< TINY_PERSON_THRESHOLD px), only Tier 1 is used
    because HSV analysis is too noisy at that scale.
    """
    x1, y1, x2, y2 = person_box
    box_h = y2 - y1
    box_w = x2 - x1

    # ── Tier 1: PPE model matching ──
    matched_hardhat, matched_no_hardhat = match_ppe_detections(
        person_box, detected_hardhats, detected_no_hardhats
    )

    if matched_hardhat:
        return HelmetVerdict(
            has_helmet=True, confidence=0.95, method="ppe_model",
            details={"tier": 1, "match": "hardhat"}
        )

    if matched_no_hardhat:
        return HelmetVerdict(
            has_helmet=False, confidence=0.90, method="ppe_model",
            details={"tier": 1, "match": "no_hardhat"}
        )

    # ── Tier 2 + 3: Color analysis (skip for tiny persons) ──
    if box_h < Config.TINY_PERSON_THRESHOLD:
        return HelmetVerdict(
            has_helmet=False, confidence=0.50, method="default",
            details={"reason": "too_small_for_color_analysis", "height_px": box_h}
        )

    is_small = box_h < Config.SMALL_PERSON_THRESHOLD

    head_roi = extract_head_region(img, x1, y1, x2, y2, img_h, img_w)
    if head_roi is None:
        return HelmetVerdict(
            has_helmet=False, confidence=0.50, method="default",
            details={"reason": "no_head_region"}
        )

    verdict = analyze_helmet_colors(head_roi, is_small_person=is_small)
    return verdict


# ══════════════════════════════════════════════════════════════════════════════
# CLI & MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-label v2: Multi-stage PPE detection pipeline"
    )
    parser.add_argument(
        "--image_dir", type=str, default=None,
        help="Path to image directory (absolute or relative to script dir)"
    )
    parser.add_argument(
        "--label_dir", type=str, default=None,
        help="Path to label output directory (default: labels_v2/)"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to YOLO11x model (default: ../yolo11x.pt)"
    )
    parser.add_argument(
        "--ppe_model_path", type=str, default=None,
        help="Path to fine-tuned PPE model (default: best.pt)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Resolve paths ──
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    if args.model_path:
        MODEL_PATH = (args.model_path if os.path.isabs(args.model_path)
                       else os.path.join(SCRIPT_DIR, args.model_path))
    else:
        MODEL_PATH = os.path.join(PROJECT_ROOT, "yolo11x.pt")

    if args.ppe_model_path:
        BEST_MODEL_PATH = (args.ppe_model_path if os.path.isabs(args.ppe_model_path)
                            else os.path.join(SCRIPT_DIR, args.ppe_model_path))
    else:
        BEST_MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")

    if args.image_dir:
        IMAGE_DIR = (args.image_dir if os.path.isabs(args.image_dir)
                     else os.path.join(SCRIPT_DIR, args.image_dir))
    else:
        IMAGE_DIR = os.path.join(SCRIPT_DIR, "images")

    if args.label_dir:
        LABEL_DIR = (args.label_dir if os.path.isabs(args.label_dir)
                     else os.path.join(SCRIPT_DIR, args.label_dir))
    else:
        LABEL_DIR = os.path.join(SCRIPT_DIR, "labels_v2")

    NAMES_FILE = os.path.join(SCRIPT_DIR, "lewl.names")

    # ── Print configuration ──
    print("=" * 65)
    print("  AUTO-LABEL v2 — Multi-Stage PPE Detection Pipeline")
    print("=" * 65)
    print(f"  Person model  : {MODEL_PATH}")
    print(f"  PPE model     : {BEST_MODEL_PATH}")
    print(f"  Image dir     : {IMAGE_DIR}")
    print(f"  Output labels : {LABEL_DIR}")
    print(f"  Names file    : {NAMES_FILE}")
    print("=" * 65)
    sys.stdout.flush()

    # ── Validate paths ──
    if not os.path.isfile(MODEL_PATH):
        print(f"[ERROR] Person model not found: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.isfile(BEST_MODEL_PATH):
        print(f"[ERROR] PPE model not found: {BEST_MODEL_PATH}")
        sys.exit(1)
    if not os.path.isdir(IMAGE_DIR):
        print(f"[ERROR] Image directory not found: {IMAGE_DIR}")
        sys.exit(1)

    # ── Read class names ──
    with open(NAMES_FILE, "r") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
    print(f"  Classes: {CLASS_NAMES}")
    print(f"    0 → '{CLASS_NAMES[0]}' (PPE / helmet)")
    print(f"    1 → '{CLASS_NAMES[1]}' (no PPE / no helmet)")
    sys.stdout.flush()

    # ── Create output directory ──
    os.makedirs(LABEL_DIR, exist_ok=True)

    # ── Load models ──
    print(f"\n  Loading person detection model...")
    sys.stdout.flush()
    person_model = YOLO(MODEL_PATH)
    print(f"  ✓ Person model loaded")

    print(f"  Loading fine-tuned PPE model...")
    sys.stdout.flush()
    ppe_model = YOLO(BEST_MODEL_PATH)
    print(f"  ✓ PPE model loaded")
    print()
    sys.stdout.flush()

    # ── Collect image files ──
    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
    total_images = len(image_files)
    print(f"  Found {total_images} images to process.\n")
    sys.stdout.flush()

    # ── Statistics ──
    stats = {
        "total_detections": 0,
        "class_p": 0,
        "class_n": 0,
        "images_processed": 0,
        "images_skipped": 0,
        "tier1_hardhat": 0,
        "tier1_no_hardhat": 0,
        "tier2_color": 0,
        "tier_default": 0,
        "colors_detected": {"yellow": 0, "white": 0, "red": 0},
        "skipped_tiny": 0,
    }

    start_time = time.time()

    # ── Progress bar ──
    iterator = enumerate(image_files)
    if HAS_TQDM:
        pbar = tqdm(
            iterator, total=total_images,
            desc="Auto-labeling v2", unit="img",
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}"
        )
    else:
        pbar = iterator

    # ══════════════════════════════════════════════════════════════════════
    # MAIN PROCESSING LOOP
    # ══════════════════════════════════════════════════════════════════════

    for idx, image_file in pbar:
        image_path = os.path.join(IMAGE_DIR, image_file)
        img = cv2.imread(image_path)
        if img is None:
            stats["images_skipped"] += 1
            continue

        img_h, img_w = img.shape[:2]
        label_filename = os.path.splitext(image_file)[0] + ".txt"
        label_filepath = os.path.join(LABEL_DIR, label_filename)

        # ── Step 1: Detect persons (YOLO11x, COCO class 0) ──
        person_results = person_model.predict(
            source=img,
            classes=[0],
            conf=Config.PERSON_CONF_THRESHOLD,
            iou=Config.PERSON_IOU_THRESHOLD,
            verbose=False
        )

        # ── Step 2: Detect hardhats / no-hardhats (PPE model) ──
        ppe_results = ppe_model.predict(
            source=img,
            classes=[0, 2],  # 0 = Hardhat, 2 = NO-Hardhat
            conf=Config.PPE_MODEL_CONF,
            iou=Config.PPE_MODEL_IOU,
            verbose=False
        )

        # Extract PPE detections
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

        # ── Step 3: Classify each person ──
        annotations = []

        for result in person_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Clamp to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)

                box_h = y2 - y1
                box_w = x2 - x1

                # Skip extremely small detections (noise)
                if box_h < Config.MIN_PERSON_HEIGHT_PX or box_w < Config.MIN_PERSON_WIDTH_PX:
                    continue

                # ── Run full pipeline ──
                verdict = classify_person(
                    img, (x1, y1, x2, y2),
                    detected_hardhats, detected_no_hardhats,
                    img_h, img_w
                )

                # ── Assign class ──
                class_id = 0 if verdict.has_helmet else 1

                # ── Convert to YOLO format ──
                x_center = (x1 + x2) / (2.0 * img_w)
                y_center = (y1 + y2) / (2.0 * img_h)
                norm_width = (x2 - x1) / float(img_w)
                norm_height = (y2 - y1) / float(img_h)

                annotations.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{norm_width:.6f} {norm_height:.6f}"
                )

                # ── Update statistics ──
                stats["total_detections"] += 1
                if class_id == 0:
                    stats["class_p"] += 1
                else:
                    stats["class_n"] += 1

                if verdict.method == "ppe_model":
                    if verdict.has_helmet:
                        stats["tier1_hardhat"] += 1
                    else:
                        stats["tier1_no_hardhat"] += 1
                elif verdict.method == "color_analysis":
                    stats["tier2_color"] += 1
                    if verdict.color_detected:
                        stats["colors_detected"][verdict.color_detected] = (
                            stats["colors_detected"].get(verdict.color_detected, 0) + 1
                        )
                else:
                    stats["tier_default"] += 1
                    if verdict.details.get("reason") == "too_small_for_color_analysis":
                        stats["skipped_tiny"] += 1

        # ── Write label file ──
        with open(label_filepath, 'w') as f:
            if annotations:
                f.write("\n".join(annotations) + "\n")

        stats["images_processed"] += 1

        # ── Update progress ──
        if HAS_TQDM:
            pbar.set_postfix({
                "det": stats['total_detections'],
                "p": stats['class_p'],
                "n": stats['class_n'],
            })
        else:
            if (idx + 1) % 50 == 0 or (idx + 1) == total_images:
                pct = (idx + 1) / total_images * 100
                bar_len = 30
                filled = int(bar_len * (idx + 1) // total_images)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(
                    f"\r  [{bar}] {pct:5.1f}% [{idx+1}/{total_images}] "
                    f"det={stats['total_detections']} "
                    f"p={stats['class_p']} n={stats['class_n']}",
                    end='', flush=True
                )
                if (idx + 1) == total_images:
                    print()

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════

    elapsed = time.time() - start_time
    fps = stats["images_processed"] / max(elapsed, 0.001)

    print("\n" + "=" * 65)
    print("  AUTO-LABELING v2 COMPLETE")
    print("=" * 65)
    print(f"  Images processed  : {stats['images_processed']}")
    print(f"  Images skipped    : {stats['images_skipped']}")
    print(f"  Total detections  : {stats['total_detections']}")
    print(f"  Class 'p' (PPE)   : {stats['class_p']}")
    print(f"  Class 'n' (no PPE): {stats['class_n']}")
    print(f"  ─────────────────────────────────────")
    print(f"  Tier 1 (PPE model)")
    print(f"    Hardhat matches : {stats['tier1_hardhat']}")
    print(f"    No-hat matches  : {stats['tier1_no_hardhat']}")
    print(f"  Tier 2 (Color analysis)")
    print(f"    Color detections: {stats['tier2_color']}")
    print(f"      Yellow helmets: {stats['colors_detected'].get('yellow', 0)}")
    print(f"      White helmets : {stats['colors_detected'].get('white', 0)}")
    print(f"      Red helmets   : {stats['colors_detected'].get('red', 0)}")
    print(f"  Default (no match): {stats['tier_default']}")
    print(f"    Skipped (tiny)  : {stats['skipped_tiny']}")
    print(f"  ─────────────────────────────────────")
    print(f"  Processing time   : {elapsed:.1f}s ({fps:.1f} img/s)")
    print(f"  Labels saved to   : {LABEL_DIR}")
    print(f"  Names file used   : {NAMES_FILE}")
    print("=" * 65)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
