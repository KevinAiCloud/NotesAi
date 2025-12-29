import cv2
import numpy as np

def preprocess_for_ocr(image_path, output_path):
    # Load image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize (OCR likes larger text)
    gray = cv2.resize(
        gray,
        None,
        fx=1.7,
        fy=1.7,
        interpolation=cv2.INTER_CUBIC
    )

    # --- Background estimation (scanner trick) ---
    # Large blur to estimate paper background
    background = cv2.GaussianBlur(gray, (61, 61), 0)

    # Normalize image by background
    normalized = cv2.divide(gray, background, scale=255)

    # --- Mild contrast enhancement ---
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(16, 16)
    )
    enhanced = clahe.apply(normalized)

    # --- Gentle smoothing (remove paper grain only) ---
    smoothed = cv2.medianBlur(enhanced, 3)

    # --- Final normalization ---
    final = cv2.normalize(
        smoothed,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX
    )

    cv2.imwrite(output_path, final)
