#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
import logging
from tqdm import tqdm
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze segmentation dataset to determine classes and weights')
    parser.add_argument('--filenames_path', type=str, required=True, 
                        help='Path to the dataset filenames file (text file with image paths)')
    parser.add_argument('--dataset_dir', type=str, required=True, 
                        help='Base directory of the dataset')
    parser.add_argument('--output_config', type=str, default='segmentation_config.json', 
                        help='Path to save the configuration JSON file')
    parser.add_argument('--sample_size', type=int, default=None, 
                        help='Number of images to sample for analysis (None for all)')
    return parser.parse_args()

def read_image(image_path, convert_rgb=True):
    """Read an image from file"""
    try:
        image = Image.open(image_path)
        if convert_rgb:
            image = image.convert("RGB")
        return np.array(image)
    except Exception as e:
        logging.error(f"Error reading image {image_path}: {e}")
        return None

def get_unique_colors(seg_image):
    """Extract unique colors from a segmentation image"""
    # Reshape to get unique colors
    if len(seg_image.shape) == 3:  # RGB image
        unique_colors = np.unique(seg_image.reshape(-1, seg_image.shape[-1]), axis=0)
    else:  # Grayscale image
        unique_colors = np.unique(seg_image)
        unique_colors = np.array([[c, c, c] for c in unique_colors])  # Convert to RGB format
    
    return [tuple(color) for color in unique_colors]

def analyze_dataset(filenames_path, dataset_dir, sample_size=None):
    """Analyze the dataset to determine classes and their pixel frequencies"""
    # Load filenames
    with open(filenames_path, "r") as f:
        filenames = [s.split() for s in f.readlines()]
    
    if sample_size is not None and sample_size < len(filenames):
        import random
        random.seed(42)  # For reproducibility
        filenames = random.sample(filenames, sample_size)
    
    logging.info(f"Analyzing {len(filenames)} images...")
    
    # First pass: Find all unique class colors
    all_colors = set()
    for file_pair in tqdm(filenames, desc="Finding unique class colors"):
        if len(file_pair) < 2:
            continue
            
        # Load segmentation image
        seg_path = os.path.join(dataset_dir, file_pair[1])
        seg_img = read_image(seg_path, convert_rgb=True)
        
        if seg_img is None:
            continue
            
        # Get unique colors
        colors = get_unique_colors(seg_img)
        all_colors.update(colors)
    
    # Convert numpy uint8 colors to standard Python tuples of integers
    all_colors_python = set()
    for color in all_colors:
        # Convert any numpy types to standard Python types
        all_colors_python.add(tuple(int(c) for c in color))
    
    # Create mapping from colors to class indices
    class_colors = sorted(list(all_colors_python))
    class_indices = {color: idx for idx, color in enumerate(class_colors)}
    num_classes = len(class_colors)
    
    logging.info(f"Found {num_classes} unique classes in the dataset")
    
    # Second pass: Count pixel frequencies per class
    class_counts = Counter()
    total_pixels = 0
    
    for file_pair in tqdm(filenames, desc="Counting class frequencies"):
        if len(file_pair) < 2:
            continue
            
        # Load segmentation image
        seg_path = os.path.join(dataset_dir, file_pair[1])
        seg_img = read_image(seg_path, convert_rgb=True)
        
        if seg_img is None:
            continue
            
        H, W = seg_img.shape[:2]
        total_pixels += H * W
        
        # Count pixels per class
        pixels = seg_img.reshape(-1, seg_img.shape[-1])
        
        for color, class_idx in class_indices.items():
            # Convert color to numpy array for comparison
            color_array = np.array(color)
            matches = np.all(pixels == color_array, axis=1)
            class_counts[class_idx] += int(np.sum(matches))  # Convert to Python int
    
    # Calculate class weights (inverse frequency)
    class_weights = []
    for i in range(num_classes):
        count = class_counts[i]
        if count == 0:
            weight = 1.0  # Assign a default weight for classes not found
        else:
            frequency = count / total_pixels
            weight = 1.0 / frequency
        class_weights.append(float(weight))
    
    # Normalize weights
    class_weights = np.array(class_weights)
    class_weights = class_weights / np.sum(class_weights) * num_classes
    
    # Prepare results - ensure all values are standard Python types, not NumPy types
    # Convert class_indices keys to strings that can be JSON serialized
    class_indices_serializable = {}
    for color, idx in class_indices.items():
        # Convert the color tuple to a string representation
        color_str = f"{color[0]}_{color[1]}_{color[2]}"
        class_indices_serializable[color_str] = int(idx)
    
    results = {
        "num_classes": int(num_classes),
        "class_colors": [[int(c) for c in color] for color in class_colors],
        "class_indices": class_indices_serializable,
        "class_weights": [float(w) for w in class_weights.tolist()],
        "class_pixel_counts": {str(i): int(class_counts[i]) for i in range(num_classes)},
        "total_analyzed_pixels": int(total_pixels)
    }
    
    return results

def main():
    args = parse_args()
    
    try:
        # Validate inputs
        if not os.path.exists(args.filenames_path):
            raise FileNotFoundError(f"Filenames file not found: {args.filenames_path}")
        
        if not os.path.exists(args.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
        
        # Analyze the dataset
        config = analyze_dataset(args.filenames_path, args.dataset_dir, args.sample_size)
        
        # Save configuration to JSON file
        try:
            with open(args.output_config, 'w') as f:
                json.dump(config, f, indent=2)
            logging.info(f"Configuration saved to {args.output_config}")
        except TypeError as e:
            # If we encounter JSON serialization issues, add extra debug info
            logging.error(f"JSON serialization error: {e}")
            problematic_keys = []
            for key, value in config.items():
                try:
                    json.dumps({key: value})
                except:
                    problematic_keys.append(f"{key} (type: {type(value)})")
            
            if problematic_keys:
                logging.error(f"Problematic keys with non-serializable values: {', '.join(problematic_keys)}")
            
            # Try with extra conversion
            logging.info("Attempting to fix non-serializable types...")
            serializable_config = json.loads(json.dumps(config, default=lambda obj: str(obj)))
            
            with open(args.output_config, 'w') as f:
                json.dump(serializable_config, f, indent=2)
            logging.info(f"Configuration saved to {args.output_config} after conversion")
        
        # Print summary
        logging.info("Dataset Analysis Summary:")
        logging.info(f"Number of classes: {config['num_classes']}")
        logging.info("Class weights:")
        for i in range(config['num_classes']):
            color = config['class_colors'][i]
            count = config['class_pixel_counts'].get(str(i), 0)
            weight = config['class_weights'][i]
            percentage = (count / config['total_analyzed_pixels']) * 100
            logging.info(f"  Class {i} (color: RGB{tuple(color)}): {percentage:.2f}% of pixels, weight: {weight:.4f}")
        
    except Exception as e:
        logging.error(f"Error analyzing dataset: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())