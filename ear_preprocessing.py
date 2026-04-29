"""
Ear Preprocessing Utilities for UERC26
=======================================
Functions for:
  1. Ear side detection & normalization (flip all ears to one canonical side)
  2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
  3. Image sharpening (unsharp mask)

HOW EAR SIDE NORMALIZATION WORKS:
---------------------------------
The ear's outer rim (helix) creates strong vertical edges on whichever side 
it faces. By computing horizontal Sobel gradients and comparing left-half vs 
right-half edge energy, we can determine which way the ear faces.

  - Right-facing ear: helix on the RIGHT side → more edge energy on the right
  - Left-facing ear:  helix on the LEFT side  → more edge energy on the left

We flip left-facing ears horizontally so ALL ears face right (canonical side).
This ensures the model doesn't waste capacity learning that left ≈ mirror(right).
"""

import cv2
import numpy as np


def detect_ear_side(image_np):
    """
    Detect whether an ear faces left or right using horizontal edge gradients.
    
    Args:
        image_np: RGB numpy array (H, W, 3)
    
    Returns:
        'left' or 'right' indicating which side the ear's outer rim faces.
    
    How it works:
        1. Convert to grayscale
        2. Apply Sobel filter in the X direction (horizontal edges)
        3. Compare absolute edge energy on left half vs right half
        4. The half with MORE edge energy has the ear's outer helix
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise before edge detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sobel in X direction — detects vertical edges (the helix is a vertical curve)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    
    h, w = gray.shape
    mid = w // 2
    
    # Compare edge energy on each half
    left_energy = np.abs(sobel_x[:, :mid]).sum()
    right_energy = np.abs(sobel_x[:, mid:]).sum()
    
    # The side with more edge energy is where the ear's outer rim is
    return 'left' if left_energy > right_energy else 'right'


def normalize_ear_side(image_np, target_side='right'):
    """
    Flip image horizontally if needed so the ear faces the target side.
    
    Args:
        image_np: RGB numpy array (H, W, 3)
        target_side: 'right' (default) — all ears will face right after this
    
    Returns:
        Flipped or original image as RGB numpy array
    
    Example:
        Subject 0001 image 01 → right-facing ear → no flip needed
        Subject 0050 image 01 → left-facing ear  → flipped to face right
    """
    detected_side = detect_ear_side(image_np)
    if detected_side != target_side:
        image_np = cv2.flip(image_np, 1)  # 1 = horizontal flip
    return image_np


def apply_clahe(image_np, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Enhances LOCAL contrast which is great for ear images because:
    - Highlights ridges and curves (helix, antihelix, tragus)
    - Handles uneven illumination from shadows (hair, lighting angle)
    
    Applied on the L channel of LAB color space to preserve color information.
    
    Args:
        image_np: RGB numpy array
        clip_limit: Contrast limiting threshold (2.0 is a good default)
        tile_grid_size: Size of contextual regions for histogram equalization
    
    Returns:
        Enhanced RGB numpy array
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE only on the L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def apply_sharpening(image_np, sigma=1.0, amount=0.5):
    """
    Apply subtle unsharp mask sharpening.
    
    Helps recover detail from blurry ear images WITHOUT amplifying too much noise.
    
    Formula: sharpened = original + amount * (original - blurred)
    
    Args:
        image_np: RGB numpy array
        sigma: Gaussian blur sigma (1.0 = subtle)
        amount: Sharpening strength (0.5 = moderate, don't go above 1.0)
    
    Returns:
        Sharpened RGB numpy array
    """
    blurred = cv2.GaussianBlur(image_np, (0, 0), sigma)
    sharpened = cv2.addWeighted(image_np, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_ear(image_np, normalize_side=True, enhance_clahe=False, sharpen=False):
    """
    Full preprocessing pipeline for a single ear image.
    
    Args:
        image_np: RGB numpy array
        normalize_side: Whether to flip ear to canonical side
        enhance_clahe: Whether to apply CLAHE
        sharpen: Whether to apply sharpening
    
    Returns:
        Preprocessed RGB numpy array
    """
    if normalize_side:
        image_np = normalize_ear_side(image_np, target_side='right')
    
    if enhance_clahe:
        image_np = apply_clahe(image_np)
    
    if sharpen:
        image_np = apply_sharpening(image_np)
    
    return image_np


# ========================
# DEMO / TEST
# ========================
if __name__ == "__main__":
    import os
    
    # Test on a few images
    test_images = [
        "UERC26_data/data/public/0001/01.png",  # Right-facing ear
        "UERC26_data/data/public/0050/01.png",  # Left-facing ear (should be flipped)
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            side = detect_ear_side(img)
            print(f"{img_path}: Detected as {side}-facing ear")
            
            # Normalize
            normalized = normalize_ear_side(img)
            new_side = detect_ear_side(normalized)
            print(f"  After normalization: {new_side}-facing")
            
            # CLAHE
            enhanced = apply_clahe(normalized)
            print(f"  CLAHE applied: shape={enhanced.shape}")
            
            # Sharpen
            sharpened = apply_sharpening(enhanced)
            print(f"  Sharpened: shape={sharpened.shape}")
            print()
