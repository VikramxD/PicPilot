from PIL import Image
import numpy as np
from logger import rich_logger as l
from ultralytics import YOLO
import cv2
from config import yolo_model








def generate_mask(image_path: str) -> np.ndarray:
    """Method to segment image
    Args:
        image (Image): input image
    Returns:
        Image: segmented image
    """
    model = YOLO(model=yolo_model)
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

def invert_mask(mask_image: np.ndarray) -> np.ndarray:
    """Method to invert mask
    Args:
        mask_image (np.ndarray): input mask image
    Returns:
        np.ndarray: inverted mask image
    """
    inverted_mask_image = cv2.bitwise_not(mask_image)
    cv2.imwrite('invert_mask.jpg', inverted_mask_image)
    return inverted_mask_image

 
if __name__ == "__main__":
    image = Image.open("../sample_data/example1.jpg")
    mask_img,orig_image = generate_mask(image_path='../sample_data/example1.jpg')
    invert_mask(mask_image=mask_img)
    
    

