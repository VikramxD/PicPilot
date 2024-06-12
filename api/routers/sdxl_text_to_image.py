import sys
sys.path.append("../scripts")  # Path of the scripts directory
import config
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import base64
from io import BytesIO
from typing import List
import uuid
from diffusers import DiffusionPipeline
import torch
from functools import lru_cache
from s3_manager import S3ManagerService
from PIL import Image
import io
from utils import accelerator
from models.sdxl_input import InputFormat
from async_batcher.batcher import AsyncBatcher
from utils import pil_to_b64_json, pil_to_s3_json
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True


device = accelerator()
router = APIRouter()




# Load the diffusion pipeline
@lru_cache(maxsize=1)
def load_pipeline(model_name, adapter_name,enable_compile:bool):
    """
    Load the diffusion pipeline with the specified model and adapter names.

    Args:
        model_name (str): The name of the pretrained model.
        adapter_name (str): The name of the adapter.

    Returns:
        DiffusionPipeline: The loaded diffusion pipeline.
    """
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    pipe.load_lora_weights(adapter_name)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe.unet.to(memory_format=torch.channels_last)
    if enable_compile is True:
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune")
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune")
    pipe.fuse_qkv_projections()
    return pipe


loaded_pipeline = load_pipeline(config.MODEL_NAME, config.ADAPTER_NAME, config.ENABLE_COMPILE)


# SDXLLoraInference class for running inference
class SDXLLoraInference(AsyncBatcher):
    """
    Class for performing SDXL Lora inference.

    Args:
        prompt (str): The prompt for generating the image.
        negative_prompt (str): The negative prompt for generating the image.
        num_images (int): The number of images to generate.
        num_inference_steps (int): The number of inference steps to perform.
        guidance_scale (float): The scale for guiding the generation process.

    Attributes:
        pipe (DiffusionPipeline): The pre-trained diffusion pipeline.
        prompt (str): The prompt for generating the image.
        negative_prompt (str): The negative prompt for generating the image.
        num_images (int): The number of images to generate.
        num_inference_steps (int): The number of inference steps to perform.
        guidance_scale (float): The scale for guiding the generation process.

    Methods:
        run_inference: Runs the inference process and returns the generated image.
    """

    def __init__(
        self,
        prompt: str,
        negative_prompt: str,
        num_images: int,
        num_inference_steps: int,
        guidance_scale: float,
        mode :str
    ) -> None:
        self.pipe = loaded_pipeline
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_images = num_images
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.mode = mode 
        

    def run_inference(self) -> str:
        """
        Runs the inference process and returns the generated image.

        Parameters:
            mode (str): The mode for returning the generated image.
                        Possible values: "b64_json", "s3_json".
                        Defaults to "b64_json".

        Returns:
            str: The generated image in the specified format.
        """
        image = self.pipe(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            negative_prompt=self.negative_prompt,
            num_images_per_prompt=self.num_images,
        ).images[0]
        
        if self.mode == "s3_json":
            s3_url = pil_to_s3_json(image,'sdxl_image')
            return s3_url
        elif self.mode == "b64_json":
            return pil_to_b64_json(image)
        else:
            raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")
        
    
class SDXLLoraBatcher(AsyncBatcher[InputFormat, dict]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipe = loaded_pipeline

    def process_batch(self, batch: List[InputFormat]) -> List[dict]:
        results = []
        for data in batch:
            try:
                images = self.pipe(
                    prompt=data.prompt,
                    num_inference_steps=data.num_inference_steps,
                    guidance_scale=data.guidance_scale,
                    negative_prompt=data.negative_prompt,
                    num_images_per_prompt=data.num_images,
                ).images
                
                for image in images:
                    if data.mode == "s3_json":
                        result = pil_to_s3_json(image, 'sdxl_image')
                    elif data.mode == "b64_json":
                        result = pil_to_b64_json(image)
                    else:
                        raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")
                    results.append(result)
            except Exception as e:
                print(f"Error in process_batch: {e}")
                raise HTTPException(status_code=500, detail="Batch inference failed")
        return results


# Endpoint for single request
@router.post("/sdxl_v0_lora_inference")
async def sdxl_v0_lora_inference(data: InputFormat):
    inference = SDXLLoraInference(
        data.prompt,
        data.negative_prompt,
        data.num_images,
        data.num_inference_steps,
        data.guidance_scale,
        data.mode,
        
        
    )
    output_json = inference.run_inference()
    return output_json


# Endpoint for batch requests

@router.post("/sdxl_v0_lora_inference/batch")
async def sdxl_v0_lora_inference_batch(data: List[InputFormat]):
    batcher = SDXLLoraBatcher(max_batch_size=64, max_queue_time=0.001)
    try:
        predictions = await batcher.process(batch=data)
        return predictions
    except Exception as e:
        print(f"Error in /sdxl_v0_lora_inference/batch: {e}")
        raise HTTPException(status_code=500, detail="Batch inference endpoint failed")
    

