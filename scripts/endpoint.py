from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from models import kandinsky_inpainting_inference
from segment_everything import extend_image, generate_mask_from_bbox, invert_mask
from video_pipeline import fetch_video_pipeline 
from diffusers.utils import load_image
from logger import rich_logger as l
from fastapi import UploadFile, File
from config import segmentation_model, detection_model,target_height, target_width, roi_scale
from PIL import Image
import io
import tempfile






app = FastAPI(title="Product Diffusion API",
              description="API for Product Diffusion", 
              version="0.1.0",
              openapi_url="/api/v1/openapi.json")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
    
)

@app.post("/api/v1/image_outpainting")
async def image_outpainting(image: UploadFile, prompt: str, negative_prompt: str,num_inference_steps:int=30):
    """
    Perform Outpainting on an image.

    Args:
        image (UploadFile): The input image file.
        prompt (str): The prompt for the outpainting.
        negative_prompt (str): The negative prompt for the outpainting.

    Returns:
        JSONResponse: The output image path.
    """
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))
    image = load_image(image)
    image = extend_image(image, target_width=target_width, target_height=target_height, roi_scale=roi_scale)
    mask_image = generate_mask_from_bbox(image, segmentation_model, detection_model)
    mask_image = Image.fromarray(mask_image)
    mask_image = invert_mask(mask_image)
    output_image = kandinsky_inpainting_inference(prompt, negative_prompt, image, mask_image,num_inference_steps=num_inference_steps)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        output_image.save(temp_file, format='JPEG')
        temp_file_path = temp_file.name
    return FileResponse(temp_file_path, media_type='image/jpeg', filename='output_image.jpg')
    



  
    