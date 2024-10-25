import io
import base64
import time
import logging
import asyncio
from typing import Dict, Any, Tuple, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from PIL import Image
from scripts.s3_manager import S3ManagerService
from scripts.flux_inference import FluxInpaintingInference
from config_settings import settings
import runpod

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

# Global instances
global_inpainter = None
global_s3_manager = None

async def initialize_services():
    """Initialize global services if not already initialized"""
    global global_inpainter, global_s3_manager
    
    if global_inpainter is None:
        logger.info("Initializing Flux Inpainting model...")
        global_inpainter = FluxInpaintingInference()
        logger.info("Flux Inpainting model initialized successfully")
        
    if global_s3_manager is None:
        logger.info("Initializing S3 manager...")
        global_s3_manager = S3ManagerService()
        logger.info("S3 manager initialized successfully")
        
    return global_inpainter, global_s3_manager

async def decode_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode and validate the incoming inpainting request asynchronously.
    """
    try:
        logger.info("Decoding inpainting request")
        inpainting_request = InpaintingRequest(**request)
        
        # Run image decoding in thread pool
        input_image_data = await asyncio.to_thread(
            base64.b64decode, inpainting_request.input_image
        )
        mask_image_data = await asyncio.to_thread(
            base64.b64decode, inpainting_request.mask_image
        )
        
        input_image = await asyncio.to_thread(
            lambda: Image.open(io.BytesIO(input_image_data))
        )
        mask_image = await asyncio.to_thread(
            lambda: Image.open(io.BytesIO(mask_image_data))
        )
        
        logger.info("Request decoded successfully")
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

async def generate_inpainting(inputs: Dict[str, Any], inpainter: FluxInpaintingInference) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """
    Perform inpainting operation using the Flux model asynchronously.
    """
    start_time = time.time()
    
    # Run inpainting in thread pool
    result_image = await asyncio.to_thread(
        inpainter.generate_inpainting,
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

async def upload_result(image: Image.Image, metadata: Dict[str, Any], s3_manager: S3ManagerService) -> Dict[str, Any]:
    """
    Upload the generated image to S3 and prepare the response asynchronously.
    """
    try:
        # Prepare image buffer
        buffered = io.BytesIO()
        await asyncio.to_thread(image.save, buffered, format="PNG")
        buffered.seek(0)
        
        # Generate unique filename and upload
        unique_filename = await asyncio.to_thread(
            s3_manager.generate_unique_file_name, "result.png"
        )
        await asyncio.to_thread(
            s3_manager.upload_file,
            io.BytesIO(buffered.getvalue()),
            unique_filename
        )
        
        # Generate signed URL
        signed_url = await asyncio.to_thread(
            s3_manager.generate_signed_url,
            unique_filename,
            exp=43200
        )
        
        return {
            "result_url": signed_url,
            "prompt": metadata["prompt"],
            "seed": metadata["seed"],
            "time_taken": metadata["time_taken"]
        }
    except Exception as e:
        logger.error(f"Error in upload_result: {e}")
        raise

async def async_generator_handler(job: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async generator handler for RunPod with progress updates.
    """
    try:
        # Initial status
        yield {"status": "starting", "message": "Initializing inpainting process"}

        # Initialize services
        inpainter, s3_manager = await initialize_services()
        yield {"status": "processing", "message": "Services initialized successfully"}

        # Decode request
        try:
            inputs = await decode_request(job['input'])
            yield {"status": "processing", "message": "Request decoded successfully"}
        except Exception as e:
            logger.error(f"Request decode error: {e}")
            yield {"status": "error", "message": f"Error decoding request: {str(e)}"}
            return

        # Generate inpainting
        try:
            yield {"status": "processing", "message": "Starting inpainting generation"}
            result_image, metadata = await generate_inpainting(inputs, inpainter)
            
            if result_image is None:
                yield {"status": "error", "message": "Failed to generate image"}
                return
                
            yield {
                "status": "processing", 
                "message": "Inpainting generated successfully",
                "completion": f"{metadata['time_taken']:.2f}s"
            }
        except Exception as e:
            logger.error(f"Inpainting error: {e}")
            yield {"status": "error", "message": f"Error during inpainting: {str(e)}"}
            return

        # Upload result
        try:
            yield {"status": "processing", "message": "Uploading result"}
            response = await upload_result(result_image, metadata, s3_manager)
            yield {"status": "processing", "message": "Result uploaded successfully"}
        except Exception as e:
            logger.error(f"Upload error: {e}")
            yield {"status": "error", "message": f"Error uploading result: {str(e)}"}
            return

        # Final response
        yield {
            "status": "completed",
            "output": response
        }

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        yield {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

# Initialize services when the service starts
print("Initializing service...")
asyncio.get_event_loop().run_until_complete(initialize_services())
print("Service initialization complete")

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": async_generator_handler,
        "return_aggregate_stream": True
    })