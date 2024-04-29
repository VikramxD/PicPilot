from transformers import pipeline
from ultralytics import YOLO
from transformers import SamModel, SamProcessor
import torch
from PIL import Image
from diffusers.utils import load_image
<<<<<<< HEAD
=======
from PIL import Image, ImageOps
import numpy as np
import torch
from diffusers import StableVideoDiffusionPipeline








>>>>>>> aaed2f5 (Refactor config.py and models.py, and add new functions in segment_everything.py and video_pipeline.py)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






<<<<<<< HEAD
def generate_mask_from_bbox(image_path):
=======



def extend_image(image, target_width, target_height, roi_scale=0.5):
    """
    Extends an image to fit within the specified target dimensions while maintaining the aspect ratio.
    
    Args:
        image (PIL.Image.Image): The image to be extended.
        target_width (int): The desired width of the extended image.
        target_height (int): The desired height of the extended image.
        roi_scale (float, optional): The scale factor applied to the resized image. Defaults to 0.5.
    
    Returns:
        PIL.Image.Image: The extended image.
    """
    original_image = image
    original_width, original_height = original_image.size
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale * roi_scale)
    new_height = int(original_height * scale * roi_scale)
    original_image_resized = original_image.resize((new_width, new_height))
    extended_image = Image.new("RGB", (target_width, target_height), "white")
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    extended_image.paste(original_image_resized, (paste_x, paste_y))
    return extended_image





def generate_mask_from_bbox(image: Image, segmentation_model: str ,detection_model) -> Image:
>>>>>>> aaed2f5 (Refactor config.py and models.py, and add new functions in segment_everything.py and video_pipeline.py)
    """
    Generates a mask from the bounding box of an image using YOLO and SAM-ViT models.

    Args:
        image_path (str): The path to the input image.

    Returns:
        numpy.ndarray: The generated mask as a NumPy array.
    """
    # Initialize YOLO and Processor
    yolo = YOLO('yolov8m.pt')
    processor = SamProcessor.from_pretrained('facebook/sam-vit-large')
    model = SamModel.from_pretrained("facebook/sam-vit-large").to(device)

    # Generate bounding boxes
    results = yolo(image_path)
    bboxes = results[0].boxes.xyxy.tolist()
    input_boxes = [[[bboxes[0]]]]

    # Process inputs
    inputs = processor(load_image(image_path), input_boxes=input_boxes, return_tensors="pt").to("cuda")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process masks
    mask = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0][0][0].numpy()
    print(mask)
    return mask


<<<<<<< HEAD
=======




def invert_mask(mask_image: Image) -> np.ndarray:
    """Method to invert mask
    Args:
        mask_image (np.ndarray): input mask image
    Returns:
        np.ndarray: inverted mask image
    """
    inverted_mask_image = ImageOps.invert(mask_image)
    return inverted_mask_image








def fetch_video_pipeline(video_model_name):
    """
    Fetches the video pipeline for image processing.

    Args:
        video_model_name (str): The name of the video model.

    Returns:
        pipe (StableVideoDiffusionPipeline): The video pipeline.

    """
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        video_model_name, torch_dtype=torch.float16, 
    )
    pipe = pipe.to('cuda')
    pipe.unet= torch.compile(pipe.unet)
    
    
    return pipe

>>>>>>> aaed2f5 (Refactor config.py and models.py, and add new functions in segment_everything.py and video_pipeline.py)
