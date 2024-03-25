from transformers import pipeline
from ultralytics import YOLO
from transformers import SamModel, SamProcessor
import torch
from PIL import Image
from diffusers.utils import load_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






def generate_mask_from_bbox(image_path):
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


