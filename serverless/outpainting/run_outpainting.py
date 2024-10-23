import io
import base64
import time
import runpod
from typing import Dict, Any
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

class RunPodOutpaintingHandler:
    """
    RunPod handler for Outpainting service.
    """
    
    def __init__(self):
        """Initialize the Outpainting handler with model on CUDA device."""
        self.device = "cuda"
        self.outpainter = Outpainter()

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode and validate the incoming request.
        
        Args:
            request: Raw request data dictionary
            
        Returns:
            Decoded request with PIL Image and parameters
            
        Raises:
            ValueError: If request is invalid or image cannot be decoded
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

    def process_request(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete RunPod job request.
        
        Args:
            job: RunPod job dictionary containing input data
            
        Returns:
            Response dictionary with results or error information
            
        Notes:
            Handles the complete pipeline:
            1. Request decoding
            2. Model inference
            3. Result encoding and S3 upload
        """
        try:
            # Decode request
            inputs = self.decode_request(job['input'])
            
            # Start timing
            start_time = time.time()
            
            # Run outpainting
            result = self.outpainter.outpaint(
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
            
            # Calculate completion time
            completion_time = time.time() - start_time
            
            # Encode and upload result
            img_str = pil_to_s3_json(result, "outpainting_image")
            
            return {
                "result": img_str,
                "completion_time": round(completion_time, 2),
                "image_resolution": f"{result.width}x{result.height}"
            }
            
        except Exception as e:
            return {"error": str(e)}

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function.
    
    Args:
        job: RunPod job dictionary
        
    Returns:
        Processed response or error information
    """
    handler = RunPodOutpaintingHandler()
    return handler.process_request(job)

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })