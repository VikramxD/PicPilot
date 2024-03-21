
from typing import List, Tuple, Dict
import torch
from PIL import Image
import numpy as np
from logger import rich_logger as l
from ultralytics import YOLO
import cv2




def convert_to_numpy_array(image: Image) -> np.ndarray:
    """Method to convert PIL image to numpy array
    Args:
        image (Image): input image
    Returns:
        np.ndarray: numpy array
    """
    return np.array(image)





def generate_mask(image_path: str) -> np.ndarray:
    """Method to segment image
    Args:
        image (Image): input image
    Returns:
        Image: segmented image
    """
    model = YOLO(model='yolov8s-seg.pt',)
    results = model(image_path)
    for result in results:
        orig_img = result.orig_img
        masks = result.masks.xy
        height, width = result.orig_img.shape[:2]
        background = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        
        for mask in masks:
           mask = mask.astype(int)
           mask_img = np.zeros_like(orig_img)
           cv2.fillPoly(mask_img, [mask], (255, 255, 255))
           mask_img = np.array(mask_img)
           orig_img = np.array(orig_img)
           
    return mask_img, orig_img

 
if __name__ == "__main__":
    image = Image.open("../sample_data/example1.jpg")
    image = image.resize((512, 512))
    image = convert_to_numpy_array(image)
    mask_image,orig_image = generate_mask(image_path='../sample_data/example1.jpg')
    
    

