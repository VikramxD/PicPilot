import io
import base64
import time
import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from PIL import Image
from scripts.s3_manager import S3ManagerService
from scripts.flux_inference import FluxInpaintingInference
from config_settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_manager = S3ManagerService()

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

class RunPodFluxInpaintingHandler:
    """
    RunPod handler for Flux Inpainting service.
    """
    
    def __init__(self):
        self.flux_inpainter = FluxInpaintingInference()
        self.device = "cuda"

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode and validate the incoming request.
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

    def process_request(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete RunPod job request.
        """
        try:
            # Decode request
            start_time = time.time()
            inputs = self.decode_request(job['input'])
            
            # Generate inpainting
            result_image = self.flux_inpainter.generate_inpainting(
                input_image=inputs["input_image"],
                mask_image=inputs["mask_image"],
                prompt=inputs["prompt"],
                seed=inputs["seed"],
                strength=inputs["strength"],
                num_inference_steps=inputs["num_inference_steps"]
            )
            
            # Prepare output
            output = {
                "image": result_image,
                "prompt": inputs["prompt"],
                "seed": inputs["seed"],
                "time_taken": time.time() - start_time
            }
            
            # Encode and upload result
            if output["image"] is None:
                return {"error": "Failed to generate image"}

            try:
                buffered = io.BytesIO()
                output["image"].save(buffered, format="PNG")
                
                unique_filename = s3_manager.generate_unique_file_name("result.png")
                s3_manager.upload_file(io.BytesIO(buffered.getvalue()), unique_filename)
                signed_url = s3_manager.generate_signed_url(unique_filename, exp=43200)
                
                return {
                    "result_url": signed_url,
                    "prompt": output["prompt"],
                    "seed": output["seed"],
                    "time_taken": output["time_taken"]
                }
            except Exception as e:
                logger.error(f"Error in encode_response: {e}")
                return {"error": str(e)}
                
        except Exception as e:
            logger.error(f"Error in process_request: {e}")
            return {"error": str(e)}

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function.
    """
    handler = RunPodFluxInpaintingHandler()
    return handler.process_request(job)

if __name__ == "__main__":
    import runpod
    runpod.serverless.start({
        "handler": handler
    })