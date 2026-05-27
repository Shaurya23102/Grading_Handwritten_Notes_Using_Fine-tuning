# preprocessing_agent.py

import cv2
import numpy as np
import logging

logger = logging.getLogger("Preprocessing")

# Target max dimension — matches what the API will use
TARGET_MAX_DIM = 1024


def _crop_to_content(img, padding=20):
    """Crop image to the bounding box of actual content (remove empty margins)."""
    # Invert for findContours (needs white content on black bg)
    inv = cv2.bitwise_not(img)
    coords = cv2.findNonZero(inv)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    # Add padding
    y1 = max(0, y - padding)
    y2 = min(img.shape[0], y + h + padding)
    x1 = max(0, x - padding)
    x2 = min(img.shape[1], x + w + padding)
    cropped = img[y1:y2, x1:x2]
    logger.info(f"Cropped to content: {img.shape[1]}x{img.shape[0]} → {cropped.shape[1]}x{cropped.shape[0]}")
    return cropped


def _resize_to_target(img, max_dim=TARGET_MAX_DIM):
    """Resize so longest side is at most max_dim."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.info(f"Resized: {w}x{h} → {new_w}x{new_h}")
    return resized


def preprocess_image(image_path, save_path="processed.jpg"):
    logger.info(f"Loading image: {image_path}")

    # Load image
    img = cv2.imread(image_path)

    # Null-check: cv2.imread returns None for invalid paths
    if img is None:
        logger.error(f"Failed to load image: {image_path}")
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]
    logger.info(f"Image loaded: {w}x{h} pixels")

    # Step 1: Resize EARLY (before expensive operations)
    img = _resize_to_target(img)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logger.info("Converted to grayscale")

    # Step 3: Denoise (now faster on smaller image)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    logger.info("Applied denoising")

    # Step 4: Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    logger.info("Applied CLAHE contrast enhancement")

    # Step 5: Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        contrast, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    logger.info("Applied adaptive thresholding")

    # Step 6: Crop to content (remove empty margins)
    processed = _crop_to_content(thresh)

    # Save as JPEG (smaller file = faster upload to API)
    cv2.imwrite(save_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 90])
    logger.info(f"Saved processed image → {save_path}")

    return save_path