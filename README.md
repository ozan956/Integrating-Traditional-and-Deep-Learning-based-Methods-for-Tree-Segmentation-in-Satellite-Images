# Integrating Traditional and Deep Learning Methods to Detect Tree Crowns in Satellite Images

This project implements a pipeline for detecting tree centers and segmenting trees in aerial or satellite images. The process involves green object segmentation, texture detection, bounding box expansion, and contour analysis. The main function uses a watershed algorithm to refine the segmentation and improve the detection of tree centers.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Directory Structure](#directory-structure)
7. [Function Details](#function-details)

## Overview
This code processes a series of input images and JSON files, detects potential tree locations, and segments trees using a combination of:
- Green object segmentation.
- Gabor filters for texture detection.
- A watershed algorithm for improved segmentation.

It saves processed images and outputs a JSON file with detected tree coordinates.

## Features
- **Segment Green Objects**: Extracts green objects from images based on HSV thresholds.
- **Texture Detection**: Applies Gabor filters to highlight texture features.
- **Watershed Segmentation**: Refines segmentation by distinguishing tree regions.
- **Bounding Box and Contour Analysis**: Matches contours to average tree sizes to identify potential tree regions.
- **Visualization**: Draws detected centers and contours on the original images.

## Dependencies
Ensure the following Python packages are installed:
- `numpy`
- `opencv-python`
- `Pillow`
- `matplotlib`
- `scikit-image`
- `scipy`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/tree-detection.git
   cd tree-detection
   ```
2. Install dependencies:
   ```bash
   pip install numpy opencv-python Pillow matplotlib scikit-image scipy
   ```

## Usage
1. Organize the input data:
   - Place the input images and JSON files in the specified directories.
2. Run the script:
   ```bash
   python tree_detection.py
   ```
3. Outputs:
   - Processed images with tree centers and contours.
   - `detected_trees_based_on_shape.json` containing detected tree coordinates.

## Directory Structure
- `base_image_path`: Contains the main input images for processing.
- `base_cropped_image_dir`: Stores cropped images from detected bounding boxes.
- `base_image_with_boxes_dir`: Stores images with bounding boxes drawn.
- `base_image_with_contours_dir`: Stores images with contours drawn.
- `base_contour_data_dir`: Contains JSON data for contours.
- `base_image_with_only_boxes_dir`: Stores images with only detected boxes.
- `base_output_img_dir`: Final processed images.

## Function Details
### `segment_trees_with_watershed(cropped_np, max_distance=25, area_threshold=1000)`
Segments the input image using a watershed algorithm after filtering based on HSV thresholds and Gabor filtering.

### `is_near_detected_centers(tree_x, tree_y, bbox_dict, threshold=40, required_nearby=5)`
Determines if a tree center is near existing detected centers.

### `matches_average_contour(contour, avg_width, avg_height, tolerance=0.2)`
Checks if a contour matches the average tree dimensions.

## Output
The script generates:
1. Visual outputs (images with bounding boxes and contours).
2. JSON files with tree detection results.