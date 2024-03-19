
import torch
import cv2
import os
from logger import rich_logger as l
from config import image_dir, mask_dir
import supervision as sv
from ultralytics import YOLO
import numpy as np
import torch
import cv2
import os
from logger import rich_logger as l
from config import image_dir, mask_dir
import supervision as sv
from ultralytics import YOLO
import numpy as np






class MaskGenerator:
    
    def __init__(self, image_folder_path: str, device: str = "cuda:0"):
        """
        Initializes the MaskGenerator object.

        Args:
            image_folder_path (str): The path to the folder containing the images.
            device (str, optional): The device to use for computation. Defaults to "cuda:0".
        """
        self.image_folder_path = image_folder_path
        
        self.device = device
        self.model = YOLO('yolov8s-seg.pt')
        self.model.to(device=self.device)
    
    def generate_masks(self):
        results = self.model(self.image_folder_path)
        for i, result in enumerate(results):
            height, width = result.orig_img.shape[:2]
            background = np.ones((height, width, 3), dtype=np.uint8) * 255
            masks = result.masks.xy
            for j, mask in enumerate(masks):
                mask = mask.astype(int)
                cv2.drawContours(background, [mask], -1, (0, 255, 0), thickness=cv2.FILLED)
            
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            cv2.imwrite(f'{mask_dir}/mask_{i}.jpg', background)
            l.info(f"Segmented image {i} saved in the mask folder.")


        
    
    
            
            
        
        
if __name__ == "__main__":
    mask_generator = MaskGenerator(image_folder_path=image_dir, device="cuda:0")
    mask_generator.generate_masks()
