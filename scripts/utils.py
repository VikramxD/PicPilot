import torch
from ultralytics import YOLO
from transformers import pipeline 
import cv2
import numpy as np


def accelerator():
    """
    Determines the device accelerator to use based on availability.

    Returns:
        str: The name of the device accelerator ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    return device




def center_scaled_roi(image_path, bg_size, scale_factor):
    """
    Center and scale the region of interest (ROI) within a background image.

    Args:
        image_path (str): The path to the original image.
        bg_size (tuple): The size (width, height) of the background image.
        scale_factor (float): The scaling factor to apply to the ROI.

    Returns:
        numpy.ndarray: The background image with the scaled ROI centered.

    """

    original_image = cv2.imread(image_path)
    height, width = original_image.shape[:2]

    # Convert the image to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store ROI coordinates
    roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0

    # Loop over the contours
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If the contour has 4 vertices, it's likely a rectangle
        if len(approx) == 4:
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(approx)
            roi_x, roi_y, roi_w, roi_h = x, y, w, h
            break

    # Calculate dimensions for the background
    bg_width, bg_height = bg_size

    # Resize the ROI based on the scale factor
    scaled_roi_w = int(roi_w * scale_factor)
    scaled_roi_h = int(roi_h * scale_factor)

    # Calculate offsets to center the scaled ROI within the background
    x_offset = (bg_width - scaled_roi_w) // 2
    y_offset = (bg_height - scaled_roi_h) // 2

    # Resize the original image
    scaled_image = cv2.resize(original_image, (scaled_roi_w, scaled_roi_h))

    # Create a blank background
    background = np.zeros((bg_height, bg_width, 3), dtype=np.uint8)

    # Place the scaled ROI onto the background
    background[y_offset:y_offset+scaled_roi_h, x_offset:x_offset+scaled_roi_w] = scaled_image

    return background

# Define dimensions for the background (larger than the ROI)
bg_width, bg_height = 800, 600

# Define the scale factor
scale_factor = 0.5  # Adjust this value as needed

# Call the function to center the scaled ROI within the background
centered_scaled_roi = center_scaled_roi('image.jpg', (bg_width, bg_height), scale_factor)

# Display the centered scaled ROI
cv2.imshow('Centered Scaled ROI', centered_scaled_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
