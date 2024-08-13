import os
import uuid
import json
import logging
from typing import Dict, Any
import numpy as np
import torch
from PIL import Image
import lightning.pytorch as pl
from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from diffusers import DiffusionPipeline
from scripts.api_utils import accelerator, pil_to_b64_json
from scripts.outpainting import ControlNetZoeDepthOutpainting
from configs.tti_settings import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL), format=settings.LOG_FORMAT)
LOGGER = logging.getLogger("sdxl_lora_inference.server")


for key, value in settings.TORCH_INDUCTOR_CONFIG.items():
    setattr(torch._inductor.config, key, value)

DEVICE = accelerator()
pl.seed_everything(42)

def load_sdxl_pipeline(model_name: str, adapter_name: str, enable_compile: bool) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(DEVICE)
    pipe.load_lora_weights(adapter_name)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe.unet.to(memory_format=torch.channels_last)
    if enable_compile:
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune")
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune")
    pipe.fuse_qkv_projections()
    return pipe

def load_outpainting_pipeline():
    return ControlNetZoeDepthOutpainting(target_size=(1024, 1024))

# Load the pipelines
loaded_sdxl_pipeline = load_sdxl_pipeline(settings.MODEL_NAME, settings.ADAPTER_NAME, settings.ENABLE_COMPILE)
loaded_outpainting_pipeline = load_outpainting_pipeline()

class SDXLLoraInference:
    def __init__(self, prompt: str, negative_prompt: str = "", num_images: int = 1,
                 num_inference_steps: int = 50, guidance_scale: float = 7.5, mode: str = "s3_json") -> None:
        self.pipe = loaded_sdxl_pipeline
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_images = num_images
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.mode = mode

    def run_inference(self) -> Dict[str, Any]:
        image = self.pipe(
            prompt=self.prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            negative_prompt=self.negative_prompt,
            num_images_per_prompt=self.num_images,
        ).images[0]
        if self.mode == "s3_json":
            return pil_to_s3_json(image, "sdxl_image")
        elif self.mode == "b64_json":
            return pil_to_b64_json(image)
        else:
            raise ValueError("Invalid mode. Supported modes are 'b64_json' and 's3_json'.")

def run_outpainting_inference(image_path: str, request: Dict[str, Any]) -> Image.Image:
    result = loaded_outpainting_pipeline.run_pipeline(
        image_path,
        controlnet_prompt=request['controlnet_prompt'],
        controlnet_negative_prompt=request['controlnet_negative_prompt'],
        controlnet_conditioning_scale=request['controlnet_conditioning_scale'],
        controlnet_guidance_scale=request['controlnet_guidance_scale'],
        controlnet_num_inference_steps=request['controlnet_num_inference_steps'],
        controlnet_guidance_end=request['controlnet_guidance_end'],
        inpainting_prompt=request['inpainting_prompt'],
        inpainting_negative_prompt=request['inpainting_negative_prompt'],
        inpainting_guidance_scale=request['inpainting_guidance_scale'],
        inpainting_strength=request['inpainting_strength'],
        inpainting_num_inference_steps=request['inpainting_num_inference_steps']
    )
    return result

@batch
def _infer_sdxl_lora(
    prompt: np.ndarray,
    negative_prompt: np.ndarray,
    num_images: np.ndarray,
    num_inference_steps: np.ndarray,
    guidance_scale: np.ndarray,
    mode: np.ndarray,
) -> Dict[str, np.ndarray]:
    inference = SDXLLoraInference(
        prompt=prompt[0].item() if prompt[0].dtype == np.object_ else prompt[0].decode('utf-8'),
        negative_prompt=negative_prompt[0].item() if negative_prompt[0].dtype == np.object_ else negative_prompt[0].decode('utf-8'),
        num_images=int(num_images[0]),
        num_inference_steps=int(num_inference_steps[0]),
        guidance_scale=float(guidance_scale[0]),
        mode=mode[0].item() if mode[0].dtype == np.object_ else mode[0].decode('utf-8'),
    )
    result = inference.run_inference()
    json_result = json.dumps(result).encode('utf-8')
    return {"output": np.array([[json_result]])}



@batch
def _infer_outpainting(
    image: np.ndarray,
    request: np.ndarray,
) -> Dict[str, np.ndarray]:
    image_data = image[0].tobytes()
    request_dict = json.loads(request[0].decode('utf-8'))
    
    # Save image to temporary file
    file_name = f"{uuid.uuid4()}.png"
    file_path = os.path.join("/tmp", file_name)
    with open(file_path, "wb") as f:
        f.write(image_data)
    
    try:
        result = run_outpainting_inference(file_path, request_dict)
        if request_dict.get('mode') == 's3_json':
            output = pil_to_s3_json(result, "outpaint_image")
        else:
            output = pil_to_b64_json(result)
        json_result = json.dumps(output).encode('utf-8')
        return {"output": np.array([[json_result]])}
    finally:
        os.remove(file_path)

def main():
    triton_config = TritonConfig(exit_on_error=True)
    with Triton(config=triton_config) as triton:
        LOGGER.info("Loading the SDXL Lora pipeline and Outpainting pipeline")

        # Bind SDXL Lora inference
        triton.bind(
            model_name="sdxl_lora",
            infer_func=_infer_sdxl_lora,
            inputs=[
                Tensor(name="prompt", dtype=np.bytes_, shape=(1,)),
                Tensor(name="negative_prompt", dtype=np.bytes_, shape=(1,)),
                Tensor(name="num_images", dtype=np.int32, shape=(1,)),
                Tensor(name="num_inference_steps", dtype=np.int32, shape=(1,)),
                Tensor(name="guidance_scale", dtype=np.float32, shape=(1,)),
                Tensor(name="mode", dtype=np.bytes_, shape=(1,)),
            ],
            outputs=[
                Tensor(name="output", dtype=np.bytes_, shape=(1,)),
            ],
            config=ModelConfig(
                max_batch_size=16,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=settings.MAX_QUEUE_DELAY_MICROSECONDS,
                ),
            ),
            strict=True,
        )

        # Bind outpainting inference
        triton.bind(
            model_name="outpainting",
            infer_func=_infer_outpainting,
            inputs=[
                Tensor(name="image", dtype=np.bytes_, shape=(1,)),
                Tensor(name="request", dtype=np.bytes_, shape=(1,)),
            ],
            outputs=[
                Tensor(name="output", dtype=np.bytes_, shape=(1,)),
            ],
            config=ModelConfig(
                max_batch_size=16,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=settings.MAX_QUEUE_DELAY_MICROSECONDS,
                ),
            ),
            strict=True,
        )

        triton.serve()

if __name__ == "__main__":
    main()