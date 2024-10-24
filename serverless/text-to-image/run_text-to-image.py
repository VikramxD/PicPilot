import runpod
import torch
from diffusers import DiffusionPipeline
from typing import Dict, Any, List
from PIL import Image
from config_settings import settings
from configs.tti_settings import tti_settings
from scripts.api_utils import pil_to_b64_json, pil_to_s3_json

device = "cuda" if torch.cuda.is_available() else "cpu"

def setup_pipeline():
    """
    Set up and optimize the SDXL pipeline with LoRA for inference.

    Returns:
        DiffusionPipeline: Optimized SDXL pipeline ready for inference
    """
    sdxl_pipeline = DiffusionPipeline.from_pretrained(
        tti_settings.MODEL_NAME,
        torch_dtype=torch.bfloat16
    ).to(device)
    
    sdxl_pipeline.load_lora_weights(tti_settings.ADAPTER_NAME)
    sdxl_pipeline.fuse_lora()
    
    sdxl_pipeline.unet.to(memory_format=torch.channels_last)
    if tti_settings.ENABLE_COMPILE:
        sdxl_pipeline.unet = torch.compile(
            sdxl_pipeline.unet,
            mode="max-autotune"
        )
        sdxl_pipeline.vae.decode = torch.compile(
            sdxl_pipeline.vae.decode,
            mode="max-autotune"
        )
    sdxl_pipeline.fuse_qkv_projections()
    return sdxl_pipeline

pipeline = setup_pipeline()

def decode_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode and validate the incoming request.
    
    Args:
        request: Raw request data containing generation parameters
        
    Returns:
        Dict[str, Any]: Processed request parameters including prompt, negative_prompt,
                       num_images, num_inference_steps, guidance_scale, and mode
    """
    return {
        "prompt": request["prompt"],
        "negative_prompt": request.get("negative_prompt", ""),
        "num_images": request.get("num_images", 1),
        "num_inference_steps": request.get("num_inference_steps", 50),
        "guidance_scale": request.get("guidance_scale", 7.5),
        "mode": request.get("mode", "s3_json")
    }

def generate_images(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate images using the SDXL pipeline.
    
    Args:
        params: Generation parameters including prompt, negative_prompt, num_images,
               num_inference_steps, and guidance_scale
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing generated images and their modes
    """
    images = pipeline(
        prompt=params["prompt"],
        negative_prompt=params["negative_prompt"],
        num_images_per_prompt=params["num_images"],
        num_inference_steps=params["num_inference_steps"],
        guidance_scale=params["guidance_scale"],
    ).images

    return [{"image": img, "mode": params["mode"]} for img in images]

def encode_response(output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Encode the generated image based on the specified mode.
    
    Args:
        output: Dictionary containing image and mode
        
    Returns:
        Dict[str, Any]: Encoded response either as S3 URL or base64 string
        
    Raises:
        ValueError: If the specified mode is not supported
    """
    mode = output["mode"]
    image = output["image"]
    
    if mode == "s3_json":
        return pil_to_s3_json(image, "sdxl_image")
    elif mode == "b64_json":
        return pil_to_b64_json(image)
    else:
        raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function for processing image generation requests.
    
    Args:
        job: RunPod job dictionary containing input parameters
        
    Returns:
        Dict[str, Any]: Generated images in requested format or error message
                       Returns single result for one image or list of results for multiple images
    """
    try:
        params = decode_request(job['input'])
        outputs = generate_images(params)
        results = [encode_response(output) for output in outputs]
        
        if len(results) == 1:
            return results[0]
        return {"results": results}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })