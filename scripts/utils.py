import torch
from ultralytics import YOLO
from transformers import SamModel,SamProcessor
import numpy as np
from PIL import Image 
from config import SEGMENTATION_MODEL_NAME


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



class ImageAugmentation:
    """
    Class for centering an image on a white background using ROI.

    Attributes:
        background_size (tuple): Size of the larger background where the image will be placed.
    """

    def __init__(self, background_size=(1920, 1080)):
        """
        Initialize ImageAugmentation class.

        Args:
            background_size (tuple, optional): Size of the larger background. Default is (1920, 1080).
        """
        self.background_size = background_size

    def center_image_on_background(self, image, roi):
        """
        Center the input image on a larger background using ROI.

        Args:
            image (numpy.ndarray): Input image.
            roi (tuple): Coordinates of the region of interest (x, y, width, height).

        Returns:
            numpy.ndarray: Image centered on a larger background.
        """
        w, h = self.background_size
        bg = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
        x, y, roi_w, roi_h = roi
        bg[(h - roi_h) // 2:(h - roi_h) // 2 + roi_h, (w - roi_w) // 2:(w - roi_w) // 2 + roi_w] = image
        return bg

    def detect_region_of_interest(self, image):
        """
        Detect the region of interest in the input image.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            tuple: Coordinates of the region of interest (x, y, width, height).
        """
        # Convert image to grayscale
        grayscale_image = np.array(Image.fromarray(image).convert("L"))
        
        # Calculate bounding box of non-zero region
        bbox = Image.fromarray(grayscale_image).getbbox()
        return bbox

def generate_bbox(image):
    """
    Generate bounding box for the input image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        tuple: Bounding box coordinates (x, y, width, height).
    """
    # Load YOLOv5 model
    model = YOLO("yolov8s.pt")
    results = model(image)
    # Get bounding box coordinates
    bbox = results[0].boxes.xyxy.int().tolist()
    return bbox

def generate_mask():
    model = SamModel.from_pretrained("SEGMENTATION_MODEL_NAMEz")
    processor = SamProcessor.from_pretrained("SEGMENTATION_MODEL_NAME")
    
    


if __name__ == "__main__":
    augmenter = ImageAugmentation()
    image_path = "/Users/vikram/Python/product_diffusion_api/sample_data/example1.jpg"
    image = np.array(Image.open(image_path).convert("RGB"))
    roi = augmenter.detect_region_of_interest(image)
    centered_image = augmenter.center_image_on_background(image, roi)
    bbox = generate_bbox(centered_image)
    print(bbox)
    


