from PIL import Image
import numpy as np
from logger import rich_logger as l
from ultralytics import YOLO
import cv2
from config import yolo_model
from pathlib import Path
import PIL.ImageOps




def generate_mask(image_path: str) -> np.ndarray:
    """Method to segment image
    Args:
        image_path (str): path to input image
    Returns:
        np.ndarray: segmented image mask
    """
    model = YOLO(model=yolo_model)  # Initialize YOLO model
    results = model(image_path)  # Perform object detection
    for result in results:
        orig_img = result.orig_img
        masks = result.masks.xy
        height, width = result.orig_img.shape[:2]
        mask_img = np.ones((height, width), dtype=np.uint8) * 255  # Initialize mask with white background
        
        for mask in masks:
            mask = mask.astype(int)
            cv2.fillPoly(mask_img, [mask], 0)  # Fill mask with detected object areas

    return mask_img

def invert_mask(mask_image: Image) -> np.ndarray:
    """Method to invert mask
    Args:
        mask_image (np.ndarray): input mask image
    Returns:
        np.ndarray: inverted mask image
    """
    inverted_mask_image =PIL.ImageOps.invert(mask_image)
    return inverted_mask_image

