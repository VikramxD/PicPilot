import torch
from ultralytics import YOLO
from transformers import SamModel, SamProcessor
import numpy as np
from PIL import Image
from config import SEGMENTATION_MODEL_NAME
import cv2
import matplotlib.pyplot as plt

def accelerator():
    """
    Determines the device accelerator to use based on availability.

    Returns:
        str: The name of the device accelerator ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class ImageAugmentation:
    """
    Class for centering an image on a white background using ROI.

    Attributes:
        target_width (int): Desired width of the extended image.
        target_height (int): Desired height of the extended image.
        roi_scale (float): Scale factor to determine the size of the region of interest (ROI) in the original image.
    """

    def __init__(self, target_width, target_height, roi_scale=0.5):
        """
        Initialize ImageAugmentation class.

        Args:
            target_width (int): Desired width of the extended image.
            target_height (int): Desired height of the extended image.
            roi_scale (float): Scale factor to determine the size of the region of interest (ROI) in the original image.
        """
        self.target_width = target_width
        self.target_height = target_height
        self.roi_scale = roi_scale

    def extend_image(self, image_path):
        """
        Extends the given image to the specified target dimensions while maintaining the aspect ratio of the original image.
        The image is centered based on the detected region of interest (ROI).

        Args:
            image_path (str): The path to the image file.

        Returns:
            PIL.Image.Image: The extended image with the specified dimensions.
        """
        # Open the original image
        original_image = cv2.imread(image_path)
        
        # Convert the image to grayscale for better edge detection
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # Perform edge detection to find contours
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (assumed to be the ROI)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate the center of the bounding box
        roi_center_x = x + w // 2
        roi_center_y = y + h // 2
        
        # Calculate the top-left coordinates of the ROI
        roi_x = max(0, roi_center_x - self.target_width // 2)
        roi_y = max(0, roi_center_y - self.target_height // 2)
        
        # Crop the ROI from the original image
        roi = original_image[roi_y:roi_y+self.target_height, roi_x:roi_x+self.target_width]
        
        # Create a new white background image with the target dimensions
        extended_image = np.ones((self.target_height, self.target_width, 3), dtype=np.uint8) * 255
        
        # Calculate the paste position for centering the ROI
        paste_x = (self.target_width - roi.shape[1]) // 2
        paste_y = (self.target_height - roi.shape[0]) // 2
        
        # Paste the ROI onto the white background
        extended_image[paste_y:paste_y+roi.shape[0], paste_x:paste_x+roi.shape[1]] = roi
        
        return Image.fromarray(cv2.cvtColor(extended_image, cv2.COLOR_BGR2RGB))


    def generate_bbox(self, image):
        """
        Generate bounding box for the input image.

        Args:
            image: The input image.

        Returns:
            list: Bounding box coordinates [x_min, y_min, x_max, y_max].
        """
        model = YOLO("yolov8s.pt")
        results = model(image)
        bbox = results[0].boxes.xyxy.tolist()
        return bbox

    def generate_mask(self, image, bbox):
        """
        Generates masks for the given image using a segmentation model.

        Args:
            image: The input image for which masks need to be generated.
            bbox: Bounding box coordinates [x_min, y_min, x_max, y_max].

        Returns:
            numpy.ndarray: The generated mask.
        """
        model = SamModel.from_pretrained(SEGMENTATION_MODEL_NAME).to(device=accelerator())
        processor = SamProcessor.from_pretrained(SEGMENTATION_MODEL_NAME)
        
        # Ensure bbox is in the correct format
        bbox_list = [bbox]  # Convert bbox to list of lists
        
        # Pass bbox as a list of lists to SamProcessor
        inputs = processor(image, input_boxes=bbox_list, return_tensors="pt").to(device=accelerator())
        with torch.no_grad():
          outputs = model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )
        
        return masks[0].cpu().numpy()

if __name__ == "__main__":
    augmenter = ImageAugmentation(target_width=1920, target_height=1080, roi_scale=0.3)
    image_path = "/home/product_diffusion_api/sample_data/example1.jpg"
    extended_image = augmenter.extend_image(image_path)
    bbox = augmenter.generate_bbox(extended_image)
    mask = augmenter.generate_mask(extended_image, bbox)
    plt.imsave('mask.jpg', mask)
    #Image.fromarray(mask).save("centered_image_with_mask.jpg")
