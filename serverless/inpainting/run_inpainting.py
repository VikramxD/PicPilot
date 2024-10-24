import io
import base64
import time
import logging
from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field
from PIL import Image
from scripts.s3_manager import S3ManagerService
from scripts.flux_inference import FluxInpaintingInference
from config_settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InpaintingRequest(BaseModel):
    """
    Model representing an inpainting request.
    """
    prompt: str = Field(..., description="The prompt for inpainting")
    strength: float = Field(0.8, ge=0.0, le=1.0, description="Strength of inpainting effect")
    seed: int = Field(42, description="Random seed for reproducibility")
    num_inference_steps: int = Field(50, ge=1, le=1000, description="Number of inference steps")
    input_image: str = Field(..., description="Base64 encoded input image")
    mask_image: str = Field(..., description="Base64 encoded mask image")

device = "cuda"
flux_inpainter = FluxInpaintingInference()
s3_manager = S3ManagerService()

def decode_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode and validate the incoming inpainting request.
    
    Args:
        request: Raw request data containing images and parameters
        
    Returns:
        Dict containing decoded images and validated parameters
        
    Raises:
        Exception: If request validation or image decoding fails
    """
    try:
        inpainting_request = InpaintingRequest(**request)
        
        input_image = Image.open(io.BytesIO(base64.b64decode(inpainting_request.input_image)))
        mask_image = Image.open(io.BytesIO(base64.b64decode(inpainting_request.mask_image)))
        
        return {
            "prompt": inpainting_request.prompt,
            "input_image": input_image,
            "mask_image": mask_image,
            "strength": inpainting_request.strength,
            "seed": inpainting_request.seed,
            "num_inference_steps": inpainting_request.num_inference_steps
        }
    except Exception as e:
        logger.error(f"Error in decode_request: {e}")
        raise

def generate_inpainting(inputs: Dict[str, Any]) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """
    Perform inpainting operation using the Flux model.
    
    Args:
        inputs: Dictionary containing input images and parameters
        
    Returns:
        Tuple containing the inpainted image and metadata
    """
    start_time = time.time()
    
    result_image = flux_inpainter.generate_inpainting(
        input_image=inputs["input_image"],
        mask_image=inputs["mask_image"],
        prompt=inputs["prompt"],
        seed=inputs["seed"],
        strength=inputs["strength"],
        num_inference_steps=inputs["num_inference_steps"]
    )
    
    output = {
        "image": result_image,
        "prompt": inputs["prompt"],
        "seed": inputs["seed"],
        "time_taken": time.time() - start_time
    }
    
    return result_image, output

def upload_result(image: Image.Image, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload the generated image to S3 and prepare the response.
    
    Args:
        image: Generated inpainting image
        metadata: Dictionary containing generation metadata
        
    Returns:
        Dict containing S3 URL and generation metadata
        
    Raises:
        Exception: If image upload or URL generation fails
    """
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        
        unique_filename = s3_manager.generate_unique_file_name("result.png")
        s3_manager.upload_file(io.BytesIO(buffered.getvalue()), unique_filename)
        signed_url = s3_manager.generate_signed_url(unique_filename, exp=43200)
        
        return {
            "result_url": signed_url,
            "prompt": metadata["prompt"],
            "seed": metadata["seed"],
            "time_taken": metadata["time_taken"]
        }
    except Exception as e:
        logger.error(f"Error in upload_result: {e}")
        raise

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function for processing inpainting requests.
    
    Args:
        job: RunPod job dictionary containing input data
        
    Returns:
        Dict containing either the processed results or error information
    """
    try:
        inputs = decode_request(job['input'])
        result_image, metadata = generate_inpainting(inputs)
        
        if result_image is None:
            return {"error": "Failed to generate image"}
            
        return upload_result(result_image, metadata)
        
    except Exception as e:
        logger.error(f"Error in handler: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import runpod
    runpod.serverless.start({
        "handler": handler
    })