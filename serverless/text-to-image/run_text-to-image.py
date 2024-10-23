import runpod
import torch
from diffusers import DiffusionPipeline
from typing import Dict, Any, List
from PIL import Image
from config_settings import settings
from configs.tti_settings import tti_settings
from scripts.api_utils import pil_to_b64_json, pil_to_s3_json

class RunPodSDXLHandler:
    """
    RunPod handler for SDXL (Stable Diffusion XL) model with LoRA.
    """
    
    def __init__(self):
        """Initialize and optimize the SDXL pipeline with LoRA."""
        self.device = "cuda"
        self.setup_pipeline()

    def setup_pipeline(self) -> None:
        """
        Set up and optimize the SDXL pipeline with LoRA for inference.
        """
        self.sdxl_pipeline = DiffusionPipeline.from_pretrained(
            tti_settings.MODEL_NAME, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        # Load and optimize LoRA
        self.sdxl_pipeline.load_lora_weights(tti_settings.ADAPTER_NAME)
        self.sdxl_pipeline.fuse_lora()
        
        # Memory and performance optimizations
        self.sdxl_pipeline.unet.to(memory_format=torch.channels_last)
        if tti_settings.ENABLE_COMPILE:
            self.sdxl_pipeline.unet = torch.compile(
                self.sdxl_pipeline.unet, 
                mode="max-autotune"
            )
            self.sdxl_pipeline.vae.decode = torch.compile(
                self.sdxl_pipeline.vae.decode, 
                mode="max-autotune"
            )
        self.sdxl_pipeline.fuse_qkv_projections()

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode and validate the incoming request.
        
        Args:
            request: Raw request data
            
        Returns:
            Processed request parameters
        """
        return {
            "prompt": request["prompt"],
            "negative_prompt": request.get("negative_prompt", ""),
            "num_images": request.get("num_images", 1),
            "num_inference_steps": request.get("num_inference_steps", 50),
            "guidance_scale": request.get("guidance_scale", 7.5),
            "mode": request.get("mode", "s3_json")
        }

    def generate_images(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate images using the SDXL pipeline.
        
        Args:
            params: Generation parameters
            
        Returns:
            List of generated images with their modes
        """
        images = self.sdxl_pipeline(
            prompt=params["prompt"],
            negative_prompt=params["negative_prompt"],
            num_images_per_prompt=params["num_images"],
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params["guidance_scale"],
        ).images

        return [{"image": img, "mode": params["mode"]} for img in images]

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encode the generated image based on the specified mode.
        
        Args:
            output: Dictionary containing image and mode
            
        Returns:
            Encoded response (S3 URL or base64)
        """
        mode = output["mode"]
        image = output["image"]
        
        if mode == "s3_json":
            return pil_to_s3_json(image, "sdxl_image")
        elif mode == "b64_json":
            return pil_to_b64_json(image)
        else:
            raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")

    def process_request(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete RunPod job request.
        
        Args:
            job: RunPod job dictionary
            
        Returns:
            Generated images in requested format
        """
        try:
            # Decode request
            params = self.decode_request(job['input'])
            
            # Generate images
            outputs = self.generate_images(params)
            
            # Encode responses
            results = [self.encode_response(output) for output in outputs]
            
            # Return single result or list based on num_images
            if len(results) == 1:
                return results[0]
            return {"results": results}
            
        except Exception as e:
            return {"error": str(e)}

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function.
    """
    handler = RunPodSDXLHandler()
    return handler.process_request(job)

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })