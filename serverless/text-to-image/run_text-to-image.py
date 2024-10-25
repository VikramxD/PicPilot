import runpod
import torch
import asyncio
import logging
from typing import Dict, Any, List, AsyncGenerator
from PIL import Image
from diffusers import DiffusionPipeline
from config_settings import settings
from configs.tti_settings import tti_settings
from scripts.api_utils import pil_to_b64_json, pil_to_s3_json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
global_pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"

async def initialize_pipeline():
    """
    Initialize and optimize the SDXL pipeline with LoRA.
    """
    global global_pipeline
    
    if global_pipeline is None:
        logger.info("Initializing SDXL pipeline...")
        
        # Run model loading in thread pool
        global_pipeline = await asyncio.to_thread(
            DiffusionPipeline.from_pretrained,
            tti_settings.MODEL_NAME,
            torch_dtype=torch.bfloat16
        )
        global_pipeline.to(device)
        
        logger.info("Loading LoRA weights...")
        await asyncio.to_thread(global_pipeline.load_lora_weights, tti_settings.ADAPTER_NAME)
        await asyncio.to_thread(global_pipeline.fuse_lora)
        
        logger.info("Optimizing pipeline...")
        global_pipeline.unet.to(memory_format=torch.channels_last)
        if tti_settings.ENABLE_COMPILE:
            global_pipeline.unet = await asyncio.to_thread(
                torch.compile,
                global_pipeline.unet,
                mode="max-autotune"
            )
            global_pipeline.vae.decode = await asyncio.to_thread(
                torch.compile,
                global_pipeline.vae.decode,
                mode="max-autotune"
            )
        await asyncio.to_thread(global_pipeline.fuse_qkv_projections)
        logger.info("Pipeline initialization complete")
    
    return global_pipeline

def decode_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Decode and validate the incoming request."""
    return {
        "prompt": request["prompt"],
        "negative_prompt": request.get("negative_prompt", ""),
        "num_images": request.get("num_images", 1),
        "num_inference_steps": request.get("num_inference_steps", 50),
        "guidance_scale": request.get("guidance_scale", 7.5),
        "mode": request.get("mode", "s3_json")
    }

async def generate_images(params: Dict[str, Any], pipeline: DiffusionPipeline) -> List[Dict[str, Any]]:
    """Generate images using the SDXL pipeline asynchronously."""
    images = await asyncio.to_thread(
        pipeline,
        prompt=params["prompt"],
        negative_prompt=params["negative_prompt"],
        num_images_per_prompt=params["num_images"],
        num_inference_steps=params["num_inference_steps"],
        guidance_scale=params["guidance_scale"],
    )
    
    return [{"image": img, "mode": params["mode"]} for img in images.images]

async def encode_response(output: Dict[str, Any]) -> Dict[str, Any]:
    """Encode the generated image asynchronously."""
    mode = output["mode"]
    image = output["image"]
    
    if mode == "s3_json":
        return await asyncio.to_thread(pil_to_s3_json, image, "sdxl_image")
    elif mode == "b64_json":
        return await asyncio.to_thread(pil_to_b64_json, image)
    else:
        raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")

async def async_generator_handler(job: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Async generator handler for RunPod with progress updates.
    """
    try:
        # Initial status
        yield {"status": "starting", "message": "Initializing image generation process"}

        # Initialize pipeline
        pipeline = await initialize_pipeline()
        yield {"status": "processing", "message": "Pipeline loaded successfully"}

        # Decode request
        try:
            params = decode_request(job['input'])
            yield {
                "status": "processing", 
                "message": "Request decoded successfully",
                "params": {
                    "prompt": params["prompt"],
                    "num_images": params["num_images"],
                    "steps": params["num_inference_steps"]
                }
            }
        except Exception as e:
            logger.error(f"Request decode error: {e}")
            yield {"status": "error", "message": f"Error decoding request: {str(e)}"}
            return

        # Generate images
        try:
            yield {"status": "processing", "message": "Generating images"}
            outputs = await generate_images(params, pipeline)
            yield {"status": "processing", "message": f"Generated {len(outputs)} images successfully"}
        except Exception as e:
            logger.error(f"Generation error: {e}")
            yield {"status": "error", "message": f"Error generating images: {str(e)}"}
            return

        # Encode responses
        try:
            yield {"status": "processing", "message": "Encoding and uploading images"}
            results = []
            for idx, output in enumerate(outputs, 1):
                result = await encode_response(output)
                results.append(result)
                yield {
                    "status": "processing", 
                    "message": f"Processed image {idx}/{len(outputs)}"
                }
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            yield {"status": "error", "message": f"Error encoding images: {str(e)}"}
            return

        # Final response
        final_response = results[0] if len(results) == 1 else {"results": results}
        yield {
            "status": "completed",
            "output": final_response
        }

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        yield {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

# Initialize pipeline at startup
logger.info("Initializing service...")
asyncio.get_event_loop().run_until_complete(initialize_pipeline())
logger.info("Service initialization complete")

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": async_generator_handler,
        "return_aggregate_stream": True
    })