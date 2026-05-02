#!/usr/bin/env python3
"""
Standalone UERC26 Dataset Precomputation Script (Resize Only)

This script processes the UERC26 dataset locally. It:
1. Scans a source directory for images.
2. Preserves the exact directory structure.
3. Resizes images to 256x256.
4. Saves the output in WebP lossless format to a target directory.

Usage:
    python standalone_precompute.py --src_dir UERC26 --dst_dir UERC26_Modified
"""

import os
import cv2
from pathlib import Path
import multiprocessing
import argparse
from tqdm import tqdm
from PIL import Image


# ==========================================
# MULTIPROCESSING WORKER
# ==========================================

def process_single_image(args):
    """
    Worker function to process a single image.
    Args:
        args: Tuple containing (src_path, dst_path)
    """
    src_path, dst_path = args

    try:
        # 1. Read image with OpenCV
        image = cv2.imread(str(src_path))
        if image is None:
            return False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Resize to target dimension
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

        # 3. Save as WebP Lossless
        os.makedirs(dst_path.parent, exist_ok=True)
        image_pil = Image.fromarray(image)
        image_pil.save(dst_path, format="WEBP", lossless=True)

        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Standalone Dataset Precomputation to WebP (Resize Only)")
    parser.add_argument('--src_dir', type=str, default=r"c:\Users\sansk\Downloads\UERC26_data(1)",
                        help='Path to the raw dataset root folder.')
    parser.add_argument('--dst_dir', type=str, default=r"c:\Users\sansk\Downloads\UERC26_Modified",
                        help='Path to save the processed WebP dataset.')
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    if not src_dir.exists():
        print(f"Error: Source directory '{src_dir}' not found!")
        print("Please ensure the dataset folder exists or provide the correct path via --src_dir.")
        return

    print(f"Scanning '{src_dir}' for images...")

    # Find all common image formats
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_paths.extend(src_dir.rglob(ext))

    # Remove duplicates caused by case-insensitive globbing on Windows
    image_paths = list(set(image_paths))

    if not image_paths:
        print(f"No images found in '{src_dir}'!")
        return

    print(f"Found {len(image_paths)} images. Preparing tasks...")

    tasks = []
    for src_path in image_paths:
        # Maintain exact directory structure
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path.with_suffix('.webp')
        tasks.append((src_path, dst_path))

    cpu_cores = multiprocessing.cpu_count()
    print(f"Starting multiprocessing pool with {cpu_cores} cores...")

    # Run multiprocessing pool with progress bar
    with multiprocessing.Pool(processes=cpu_cores) as pool:
        results = list(tqdm(pool.imap_unordered(process_single_image, tasks),
                            total=len(tasks), desc="Resizing"))

    success_count = sum(results)

    print("\n" + "="*50)
    print("PRECOMPUTATION COMPLETE!")
    print(f"Successfully processed: {success_count} / {len(tasks)} images.")
    print(f"Modified dataset saved to: {dst_dir.absolute()}")
    print("="*50)


if __name__ == "__main__":
    # Required for Windows multiprocessing compatibility
    multiprocessing.freeze_support()
    main()
