import io
import base64
import time
import runpod
from typing import Dict, Any, Tuple
from PIL import Image
from pydantic import BaseModel, Field
from scripts.outpainting import Outpainter
from scripts.api_utils import pil_to_s3_json

class OutpaintingRequest(BaseModel):
    """
    Pydantic model representing a request for outpainting inference.
    """
    image: str = Field(..., description="Base64 encoded input image")
    width: int = Field(1024, description="Target width")
    height: int = Field(1024, description="Target height")
    overlap_percentage: int = Field(10, description="Mask overlap percentage")
    num_inference_steps: int = Field(8, description="Number of inference steps")
    resize_option: str = Field("Full", description="Resize option")
    custom_resize_percentage: int = Field(100, description="Custom resize percentage")
    prompt_input: str = Field("", description="Prompt for generation")
    alignment: str = Field("Middle", description="Image alignment")
    overlap_left: bool = Field(True, description="Apply overlap on left side")
    overlap_right: bool = Field(True, description="Apply overlap on right side")
    overlap_top: bool = Field(True, description="Apply overlap on top side")
    overlap_bottom: bool = Field(True, description="Apply overlap on bottom side")

device = "cuda"
outpainter = Outpainter()

def decode_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode and validate the incoming request.
    
    Args:
        request: Raw request data containing image and parameters
        
    Returns:
        Dict containing decoded PIL Image and validated parameters
        
    Raises:
        ValueError: If request validation fails or image decoding fails
    """
    try:
        outpainting_request = OutpaintingRequest(**request)
        image_data = base64.b64decode(outpainting_request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        
        return {
            'image': image,
            'params': outpainting_request.model_dump()
        }
    except Exception as e:
        raise ValueError(f"Invalid request: {str(e)}")

def perform_outpainting(inputs: Dict[str, Any]) -> Tuple[Image.Image, float]:
    """
    Perform outpainting operation on the input image.
    
    Args:
        inputs: Dictionary containing image and outpainting parameters
        
    Returns:
        Tuple containing the outpainted image and processing time in seconds
    """
    start_time = time.time()
    
    result = outpainter.outpaint(
        inputs['image'],
        inputs['params']['width'],
        inputs['params']['height'],
        inputs['params']['overlap_percentage'],
        inputs['params']['num_inference_steps'],
        inputs['params']['resize_option'],
        inputs['params']['custom_resize_percentage'],
        inputs['params']['prompt_input'],
        inputs['params']['alignment'],
        inputs['params']['overlap_left'],
        inputs['params']['overlap_right'],
        inputs['params']['overlap_top'],
        inputs['params']['overlap_bottom']
    )
    
    completion_time = time.time() - start_time
    return result, completion_time

def format_response(result: Image.Image, completion_time: float) -> Dict[str, Any]:
    """
    Format the outpainting result for API response.
    
    Args:
        result: Outpainted PIL Image
        completion_time: Processing time in seconds
        
    Returns:
        Dict containing S3 URL, completion time, and image resolution
    """
    img_str = pil_to_s3_json(result, "outpainting_image")
    
    return {
        "result": img_str,
        "completion_time": round(completion_time, 2),
        "image_resolution": f"{result.width}x{result.height}"
    }

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function for processing outpainting requests.
    
    Args:
        job: RunPod job dictionary containing input data
        
    Returns:
        Dict containing either the processed results or error information
    """
    try:
        inputs = decode_request(job['input'])
        result, completion_time = perform_outpainting(inputs)
        return format_response(result, completion_time)
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })