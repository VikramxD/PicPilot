import torch
from ultralytics import YOLO
from transformers import SamModel, SamProcessor
import numpy as np
from PIL import Image, ImageOps
from config import SEGMENTATION_MODEL_NAME, DETECTION_MODEL_NAME

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

    def __init__(self, target_width, target_height, roi_scale=0.6):
        self.target_width = target_width
        self.target_height = target_height
        self.roi_scale = roi_scale

    def extend_image(self, image: Image) -> Image:
        """
        Extends an image to fit within the specified target dimensions while maintaining the aspect ratio.
        """
        original_width, original_height = image.size
        scale = min(self.target_width / original_width, self.target_height / original_height)
        new_width = int(original_width * scale * self.roi_scale)
        new_height = int(original_height * scale * self.roi_scale)
        resized_image = image.resize((new_width, new_height))
        extended_image = Image.new("RGB", (self.target_width, self.target_height), "white")
        paste_x = (self.target_width - new_width) // 2
        paste_y = (self.target_height - new_height) // 2
        extended_image.paste(resized_image, (paste_x, paste_y))
        return extended_image

    def generate_mask_from_bbox(self, image: Image) -> np.ndarray:
        """
        Generates a mask from the bounding box of an image using YOLO and SAM-ViT models.
        """
        yolo = YOLO(DETECTION_MODEL_NAME)
        processor = SamProcessor.from_pretrained(SEGMENTATION_MODEL_NAME)
        model = SamModel.from_pretrained(SEGMENTATION_MODEL_NAME).to(accelerator())
        
        # Run YOLO detection
        results = yolo(np.array(image))
        bboxes = results[0].boxes.xyxy.tolist()
        print(bboxes)
        
        
        # Prepare inputs for SAM
        inputs = processor(image, input_boxes=[bboxes], return_tensors="pt").to(device=accelerator())
        with torch.no_grad():
            outputs = model(**inputs)
            masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        
        
        return masks[0].numpy()

    def invert_mask(self, mask_image: np.ndarray) -> np.ndarray:
        """
        Inverts the given mask image.
        """
        mask_image = (mask_image * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_image)
        
        inverted_mask_pil = ImageOps.invert(mask_pil.convert("L"))
        return inverted_mask_pil

if __name__ == "__main__":
    augmenter = ImageAugmentation(target_width=1920, target_height=1080, roi_scale=0.6)
    image_path = "/home/product_diffusion_api/sample_data/example1.jpg"
    image = Image.open(image_path)
    extended_image = augmenter.extend_image(image)
    mask = augmenter.generate_mask_from_bbox(extended_image)
    