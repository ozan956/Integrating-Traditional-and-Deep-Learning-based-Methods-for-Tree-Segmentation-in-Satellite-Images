import os
import cv2
import json
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from scipy.ndimage import maximum_filter, label

# Function to retrieve the image ID by filename from a JSON file
def get_image_id_by_filename(json_file_path, target_filename):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        images = data.get('images', [])
        for image in images:
            if image.get('file_name') == target_filename:
                return image.get('id')
        
        print(f"Filename '{target_filename}' not found in the JSON data.")
        return None
    except FileNotFoundError:
        print(f"JSON file '{json_file_path}' not found.")
        return None

# Function to apply Gabor filter
def apply_gabor_filter(img, ksize, sigma, theta, lamda, gamma, psi):
    g_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
    return cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

# Function for green color segmentation
def segment_green_objects(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image, mask

# Function to append new data to a JSON file
def append_to_json_file(file_path, new_data):
    try:
        with open(file_path, 'r+') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
            
            if isinstance(data, list):
                data.append(new_data)
            else:
                raise ValueError("JSON root must be a list to append data.")

            file.seek(0)
            json.dump(data, file, indent=4)
            file.truncate()
    except FileNotFoundError:
        with open(file_path, 'w') as file:
            json.dump([new_data], file, indent=4)

# Function to merge close local maxima
def merge_close_local_maxima(local_maxima, max_distance=25):
    merged_maxima = []
    for candidate in local_maxima:
        for idx, existing in enumerate(merged_maxima):
            if np.linalg.norm(np.array(candidate) - np.array(existing)) <= max_distance:
                merged_maxima[idx] = tuple(np.mean([existing, candidate], axis=0))
                break
        else:
            merged_maxima.append(candidate)
    return merged_maxima

# Define paths
GROUND_JSON = 'C:/wbf_test/beril-ozan-cem-work/new_idea/beril_work_test/ground.json'
INPUT_FOLDER = 'C:/wbf_test/beril-ozan-cem-work/new_idea/beril_work_test/inputs'
OUTPUT_FOLDER = 'C:/wbf_test/beril-ozan-cem-work/new_idea/beril_work_test/outputs'
JSON_FILE = 'C:/wbf_test/beril-ozan-cem-work/new_idea/beril_work_test/tree_centers.json'
GABOR_FOLDER = 'C:/wbf_test/beril-ozan-cem-work/new_idea/beril_work_test/output_gabor'
COLOR_FOLDER = 'C:/wbf_test/beril-ozan-cem-work/new_idea/beril_work_test/output_color'

# Ensure output directories exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(GABOR_FOLDER, exist_ok=True)
os.makedirs(COLOR_FOLDER, exist_ok=True)

# Main processing loop
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Load image
        img_path = os.path.join(INPUT_FOLDER, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Step 1: Green Color Segmentation
        color_segmented_image, color_mask = segment_green_objects(img)

        # Step 2: Gabor Filter
        ksize, sigma, theta, lamda, gamma, psi = 3, 2.0, np.pi / 4, 3.0, 0.5, 0
        gabor_channels = [apply_gabor_filter(img[:, :, ch], ksize, sigma, theta, lamda, gamma, psi) for ch in range(3)]
        gabor_combined = cv2.merge(gabor_channels)
        gabor_gray = cv2.cvtColor(gabor_combined, cv2.COLOR_RGB2GRAY)
        _, gabor_mask = cv2.threshold(gabor_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gabor_mask = cv2.bitwise_not(gabor_mask)

        # Save intermediate results
        cv2.imwrite(os.path.join(GABOR_FOLDER, f'processed_{filename}'), gabor_mask)
        cv2.imwrite(os.path.join(COLOR_FOLDER, f'processed_{filename}'), color_mask)

        # Step 3: Joint Probability Map
        joint_prob = 0.5 * (color_mask.astype('float32') / 255.0) + 0.5 * (gabor_mask.astype('float32') / 255.0)
        joint_prob = (255 * cv2.morphologyEx(joint_prob, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))).astype(np.uint8)
        _, thresholded_image = cv2.threshold(joint_prob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 4: Watershed Segmentation
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)

        watershed_filled = np.zeros_like(img)
        img[markers == -1] = [255, 0, 0]
        for marker in np.unique(markers):
            if marker > 1:
                watershed_filled[markers == marker] = [255, 0, 0]

        # Step 5: Find and Merge Local Maxima
        local_maxima = maximum_filter(dist_transform, size=5) == dist_transform
        labeled_maxima, num_features = label(local_maxima)

        filtered_coordinates = [
            tuple(coord)
            for i in range(1, num_features + 1)
            if np.sum(labeled_maxima == i) < 1000  # Threshold to filter large areas
            for coord in np.argwhere(labeled_maxima == i)
        ]
        merged_maxima = merge_close_local_maxima(filtered_coordinates)

        # Save tree centers and annotate image
        tree_no = 0
        for x, y in merged_maxima:
            tree_no += 1
            tree_data = {
                "id": get_image_id_by_filename(GROUND_JSON, filename),
                "file_name": filename,
                "tree_no": tree_no,
                "tree_center_x": int(x),
                "tree_center_y": int(y),
            }
            append_to_json_file(JSON_FILE, tree_data)
            cv2.circle(watershed_filled, (int(y), int(x)), 4, (255, 255, 255), -1)

        output_image_path = os.path.join(OUTPUT_FOLDER, f'processed_{filename}')
        cv2.imwrite(output_image_path, cv2.cvtColor(watershed_filled, cv2.COLOR_RGB2BGR))

        print(f"Processed: {filename}")
