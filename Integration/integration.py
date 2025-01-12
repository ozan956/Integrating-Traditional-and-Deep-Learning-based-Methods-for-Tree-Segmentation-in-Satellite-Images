import json
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.ndimage import maximum_filter, label

# Define the base paths
base_image_path = 'C:/wbf_test/beril-ozan-cem-work/new_idea/watershred/'
base_cropped_image_dir = 'C:/wbf_test/beril-ozan-cem-work/new_idea/cropped_image/'
base_image_with_boxes_dir = 'C:/wbf_test/beril-ozan-cem-work/new_idea/images_with_boxes/'
base_image_with_contours_dir = 'C:/wbf_test/beril-ozan-cem-work/new_idea/images_with_contours/'
base_contour_data_dir = 'C:/wbf_test/beril-ozan-cem-work/new_idea/contour_data/'
base_image_with_only_boxes_dir = 'C:/wbf_test/beril-ozan-cem-work/new_idea/images_with_only_boxes/'
origin_base_dir = 'C:/wbf_test/beril-ozan-cem-work/new_idea/beril_work_test/inputs/'
base_output_img_dir = 'C:/wbf_test/beril-ozan-cem-work/new_idea/resultant-new-version-all/' 

def segment_trees_with_watershed(cropped_np, max_distance=25, area_threshold=1000):

    #plt.figure(figsize=(10, 5))
    #plt.subplot(1, 1, 1)
    #plt.imshow(cropped_np)
    #plt.title("Original Mask")
    #plt.axis('off')
    #plt.show()

    # Segment green objects in the RGB input image
    def segment_green_objects(image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        segmented_image = cv2.bitwise_and(image, image, mask=mask)
        return segmented_image, mask

    color_segmented_image, color_mask = segment_green_objects(cropped_np)

    # Apply Gabor filter for texture detection on each channel
    def apply_gabor_filter(img, ksize=3, sigma=2.0, theta=np.pi/4, lamda=3.0, gamma=0.5, psi=0):
        g_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
        return filtered_img

    # Filter each RGB channel and combine
    filtered_r = apply_gabor_filter(cropped_np[:, :, 0])
    filtered_g = apply_gabor_filter(cropped_np[:, :, 1])
    filtered_b = apply_gabor_filter(cropped_np[:, :, 2])
    combined_filtered = cv2.merge([filtered_r, filtered_g, filtered_b])
    gray_filtered = cv2.cvtColor(combined_filtered, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gabor_mask = cv2.bitwise_not(binary_mask)

    # Joint probability map
    joint_probability = (0.5 * color_mask.astype('float32') / 255.0 + 0.5 * gabor_mask.astype('float32') / 255.0)
    kernel = np.ones((9, 9), np.uint8)
    joint_probability = cv2.morphologyEx(joint_probability, cv2.MORPH_OPEN, kernel)
    joint_probability = (joint_probability * 255).astype(np.uint8)



    (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(binary_mask)
    thresholded_image2 = 255-binary_mask*255
    thresholded_image2 = binary_mask.astype(np.uint8)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresholded_image2,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.01*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(cropped_np,markers)

    # Create an empty image to hold the segmented regions
    watershed_filled = np.zeros_like(cropped_np)

    cropped_np[markers == -1] = [255,0,0]

    # Fill each region with red
    for marker in np.unique(markers):
        if marker == -1:
            # Boundary marker, leave as it is (black)
            continue
        elif marker > 1:  # Ignore the background
            # Fill inside the regions corresponding to each unique marker with red
            watershed_filled[markers == marker] = [255, 0, 0]  # Red color for the region

    # Make sure the background is black (where marker is 1)
    watershed_filled[markers == 1] = [0, 0, 0]

    # Display the result
    #plt.figure(figsize=(10, 5))
    #plt.subplot(1, 2, 1)
    #plt.imshow(markers, cmap='gray')
    #plt.title('Watershed Markers')
    #plt.axis('off')

    #plt.subplot(1, 2, 2)
    #plt.imshow(watershed_filled)
    #plt.title('Filled Segments (Watershed)')
    #plt.axis('off')
    #plt.show()


    #plt.figure(figsize=(10, 5))
    #plt.subplot(1, 1, 1)
    #plt.imshow(joint_probability)
    #plt.title("Original Mask")
    #plt.axis('off')
    #plt.show()

    return joint_probability

# Load JSON data
with open('C:/wbf_test/beril-ozan-cem-work/new_idea/best_fuse.json', 'r') as f:
    correct_data = json.load(f)
with open('C:/wbf_test/beril-ozan-cem-work/new_idea/tree_centers.json', 'r') as f:
    tree_centers = json.load(f)
with open('C:/wbf_test/beril-ozan-cem-work/new_idea/probab_maps_id.json', 'r') as f:
    probab_maps = json.load(f)

# Define color for blue regions
blue_color_lower = np.array([100, 0, 0])  # Lower bound for blue in BGR
blue_color_upper = np.array([255, 50, 50])  # Upper bound for blue in BGR

# Define functions
def is_near_detected_centers(tree_x, tree_y, bbox_dict, threshold=40, required_nearby=5):
    nearby_count = 0
    check_centers = set()
    for bbox in bbox_dict:
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        check_centers.add(((x_min + x_max)/2, (y_min + y_max)/2))

    for (det_x, det_y) in check_centers:
        distance = np.sqrt((tree_x - det_x) ** 2 + (tree_y - det_y) ** 2)
        if distance <= threshold:
            nearby_count += 1
        if nearby_count >= required_nearby:
            return True
    return False

def matches_average_contour(contour, avg_width, avg_height, tolerance=0.2):
    x, y, w, h = cv2.boundingRect(contour)
    return (avg_width * (1 - tolerance) <= w <= avg_width * (1 + tolerance) and
            avg_height * (1 - tolerance) <= h <= avg_height * (1 + tolerance))

# Main processing loop
for entry in probab_maps:
    image_id = entry['id']
    bbox_dict = []
    detected_centers = set()
    contour_heights = []
    contour_widths = []

    image_path = entry['file_name']
    image_path_R = os.path.join(base_image_path, image_path)
    image_path_original  = os.path.join(origin_base_dir, image_path)

    cropped_image_dir = os.path.join(base_cropped_image_dir, str(image_id))
    os.makedirs(cropped_image_dir, exist_ok=True)
    os.makedirs(base_image_with_boxes_dir, exist_ok=True)
    os.makedirs(base_image_with_contours_dir, exist_ok=True)
    os.makedirs(base_contour_data_dir, exist_ok=True)
    os.makedirs(base_image_with_only_boxes_dir, exist_ok=True)

    original_image = Image.open(image_path_R).convert("RGB")
    draw = ImageDraw.Draw(original_image)  # Drawing object on the image

    # Collect bounding boxes with scores above the threshold
    for detected in correct_data:
        if detected['score'] < 0.8:
            continue
        if detected['image_id'] == image_id:
            bbox_dict.append(detected['bbox'])
    num_of_bbox = 0
    num_of_centers = 0
    for bbox in bbox_dict:
        num_of_bbox = num_of_bbox + 1
        x_min, y_min, width, height = bbox
        
        # Expand the bounding box by a certain amount (e.g., 10 pixels)
        padding = 10
        x_min_expanded = max(0, x_min - padding)
        y_min_expanded = max(0, y_min - padding)
        x_max_expanded = min(original_image.width, x_min + width + padding)
        y_max_expanded = min(original_image.height, y_min + height + padding)
        
        # Crop image with expanded box
        crop_box = (x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded)
        cropped_image = original_image.crop(crop_box)
        
        # Convert to grayscale and threshold
        cropped_np = np.array(cropped_image.convert("L"))
        _, thresholded = cv2.threshold(cropped_np, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            contour_widths.append(w)
            contour_heights.append(h)

    avg_width = np.mean(contour_widths) if contour_widths else 0
    avg_height = np.mean(contour_heights) if contour_heights else 0

    image_with_only_boxes = np.array(original_image)

    # Check tree centers for bounding box or contour match
    count = 0
    for center in tree_centers:

        if center['id'] == image_id:
            tree_x, tree_y = center['tree_center_x'], center['tree_center_y']
            detected = False
            for bbox in bbox_dict:
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height
                detected_centers.add(((x_min + x_max)/2, (y_min + y_max)/2))
                if (x_min - 10) <= tree_x <= (x_max+10) and (y_min-10) <= tree_y <= (y_max+10):
                    #detected_centers.add(((x_min + x_max)/2, (y_min + y_max)/2))
                    detected = True
                    break

            if not detected:
                if is_near_detected_centers(tree_x, tree_y, bbox_dict, threshold=180, required_nearby=3):
                    #print(f"Tree center at ({tree_x}, {tree_y}) detected because there is neighbour.")
                    detected_centers.add((tree_x, tree_y))
                else:
                    tree_crop = original_image.crop((tree_x - 25, tree_y - 25, tree_x + 25, tree_y + 25))
                    tree_np = np.array(tree_crop.convert("L"))
                    _, tree_thresh = cv2.threshold(tree_np, 50, 255, cv2.THRESH_BINARY)
                    tree_contours, _ = cv2.findContours(tree_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in tree_contours:
                        if matches_average_contour(contour, avg_width, avg_height, tolerance=0.02):
                            #print("burdayim")
                            detected_centers.add((tree_x, tree_y))
                            detected = True
                            break

                    if not detected:
                        #print(f"Tree center at ({tree_x}, {tree_y}) in image {image_id} is undetected.")
                        cv2.circle(image_with_only_boxes, (int(tree_x), int(tree_y)), 4, (0, 0, 0), -1)
                        mask = cv2.inRange(image_with_only_boxes, blue_color_lower, blue_color_upper)
                        kernel = np.ones((3, 3), np.uint8)
                        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
                        contours, _ = cv2.findContours(mask_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            if x <= tree_x <= x + w and y <= tree_y <= y + h:
                                cv2.drawContours(image_with_only_boxes, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)
                                #print(f"Removed blue region surrounding fake tree center at ({tree_x}, {tree_y}) in image {image_id}")


    original_image = Image.open(image_path_R).convert("RGB")
    gercek_img = Image.open(image_path_original).convert("RGB")
    original_image_np = np.array(original_image)
    for bbox in bbox_dict:
        x_min, y_min, width, height = bbox
        
        # Expand the bounding box by a certain amount (e.g., 10 pixels)
        padding = 10
        x_min_expanded = max(0, x_min - padding)
        y_min_expanded = max(0, y_min - padding)
        x_max_expanded = min(original_image.width, x_min + width + padding)
        y_max_expanded = min(original_image.height, y_min + height + padding)
        
        # Crop image with expanded box
        crop_box = (x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded)
        cropped_image = original_image.crop(crop_box)

        # Convert to grayscale and threshold
        cropped_np = np.array(cropped_image.convert("L"))
        _, thresholded = cv2.threshold(cropped_np, 50, 255, cv2.THRESH_BINARY)
        # Find contours
        contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            #print("Burdasiniz")
            x_min, y_min, width, height = bbox
            
            # Expand the bounding box by a certain amount (e.g., 10 pixels)
            padding = 5
            x_min_expanded = max(0, x_min - padding)
            y_min_expanded = max(0, y_min - padding)
            x_max_expanded = min(original_image.width, x_min + width + padding)
            y_max_expanded = min(original_image.height, y_min + height + padding)
            
            # Crop image with expanded box
            crop_box = (x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded)
            cropped_image = gercek_img.crop(crop_box)
            # Convert to grayscale and threshold
            cropped_np = np.array(cropped_image)
            watershed_result = segment_trees_with_watershed(cropped_np)

            _, thresholded = cv2.threshold(watershed_result, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            contour = contour + np.array([x_min_expanded, y_min_expanded])  # Shift contour coordinates
            contour = contour.astype(np.int32)  # Ensure integer format

            x, y, w, h = cv2.boundingRect(contour)
            contour_widths.append(w)
            contour_heights.append(h)

            # Draw contours with correct format
            cv2.drawContours(image_with_only_boxes, [contour], -1, (255, 0, 0), thickness=cv2.FILLED)


    data = []
    i = 0

    colored_img = cv2.imread(image_path_original)
    original_image_np = np.array(colored_img)

    # Display the original mask
    #plt.figure(figsize=(10, 5))
    #plt.subplot(1, 1, 1)
    #plt.imshow(image_with_only_boxes, cmap='gray')
    #plt.title("Original Mask")
    #plt.axis('off')
    #plt.show()
    
    detected_centers = list(set(detected_centers))
    for (tree_x, tree_y) in detected_centers:
        num_of_centers = num_of_centers +1
        i += 1
        mask = cv2.inRange(image_with_only_boxes, blue_color_lower, blue_color_upper)
        kernel = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=3)
        contours, _ = cv2.findContours(mask_dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # If contours are found, draw each on the original image
        for contour in contours:
            contour_array = np.array(contour, dtype=np.int32)
            cv2.drawContours(original_image_np, [contour_array], -1, (0, 255, 0), 2)

        cv2.circle(original_image_np, (int(tree_x), int(tree_y)), 4, (255, 255, 255), -1)
        entry = {
            "image_id": image_id,
            "tree_x": tree_x,
            "tree_y": tree_y,
        }
        data.append(entry)

    # Save the modified images and detected centers
    image_with_only_boxes_path = os.path.join(base_output_img_dir, f'{image_path}.jpg')
    cv2.imwrite(image_with_only_boxes_path, original_image_np)

    #print(f"Image {image_id}: Detected {len(detected_centers)} tree centers")

    # Load existing data (if any) from the file
    try:
        with open('detected_trees_based_on_shape.json', 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is empty, start with an empty list or dictionary
        existing_data = []

    # Add your new data (assuming `data` is the new information to append)
    existing_data.append(data)  # Add `data` to the list

    # Write the updated data back to the file
    with open('detected_trees_based_on_shape.json', 'w') as f:
        json.dump(existing_data, f, indent=4)
