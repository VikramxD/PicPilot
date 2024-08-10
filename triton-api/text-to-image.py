"""
Triton Inference Server Script for SDXL Lora Model

This script sets up and runs a Triton Inference Server for the SDXL Lora model.
It handles image generation requests, processes them through the SDXL Lora pipeline,
and returns the results. The server supports batch processing and can output
results in either base64-encoded JSON or by uploading to S3.

The script uses configuration settings from the 'configs.tti_settings' module and
utility functions from the 'scripts.api_utils' module.

Main components:
1. SDXLLoraInference: Class for running inference on the SDXL Lora model.
2. load_pipeline: Function to load and set up the diffusion pipeline.
3. _infer_fn: Main inference function that processes batch requests.
4. main: Sets up and starts the Triton Inference Server.

Usage:
    Run this script to start the Triton Inference Server for SDXL Lora model.
    The server will be ready to accept inference requests as configured.
"""

import logging
import json
from functools import lru_cache
from typing import List, Dict, Any

import numpy as np
import torch
from diffusers import DiffusionPipeline
from pytriton.decorators import batch
from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

from configs.tti_settings import settings
from scripts.api_utils import accelerator, pil_to_b64_json, pil_to_s3_json

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL), format=settings.LOG_FORMAT
)
LOGGER = logging.getLogger("sdxl_lora_inference.server")

# Configure torch inductor
for key, value in settings.TORCH_INDUCTOR_CONFIG.items():
    setattr(torch._inductor.config, key, value)

# Set device
DEVICE = accelerator()


@lru_cache(maxsize=1)
def load_pipeline(
    model_name: str, adapter_name: str, enable_compile: bool
) -> DiffusionPipeline:
    """
    Load and set up the diffusion pipeline with the specified model and adapter.

    Args:
        model_name (str): The name or path of the pretrained model.
        adapter_name (str): The name or path of the LoRA adapter.
        enable_compile (bool): Whether to compile the model for optimization.

    Returns:
        DiffusionPipeline: The loaded and configured diffusion pipeline.
    """
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(
        DEVICE
    )
    pipe.load_lora_weights(adapter_name)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe.unet.to(memory_format=torch.channels_last)
    if enable_compile:
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune")
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune")
    pipe.fuse_qkv_projections()
    return pipe


# Load the pipeline
loaded_pipeline = load_pipeline(
    settings.MODEL_NAME, settings.ADAPTER_NAME, settings.ENABLE_COMPILE
)


class SDXLLoraInference:
    """
    Class for running inference on the SDXL Lora model.

    This class encapsulates the inference process, including prompt processing
    and image generation using the loaded diffusion pipeline.
    """

    def __init__(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        mode: str = "s3_json",
    ) -> None:
        """
        Initialize the SDXLLoraInference instance.

        Args:
            prompt (str): The main prompt for image generation.
            negative_prompt (str, optional): Negative prompt to guide generation. Defaults to "".
            num_images (int, optional): Number of images to generate. Defaults to 1.
            num_inference_steps (int, optional): Number of denoising steps. Defaults to 50.
            guidance_scale (float, optional): Guidance scale for generation. Defaults to 7.5.
            mode (str, optional): Output mode ('s3_json' or 'b64_json'). Defaults to "s3_json".
        """
        self.pipe = loaded_pipeline
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_images = num_images
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.mode = mode

    def run_inference(self) -> Dict[str, Any]:
        """
        Run the inference process to generate images.

        Returns:
            Dict[str, Any]: A dictionary containing the generated image data or S3 URL.

        Raises:
            ValueError: If an invalid output mode is specified.
        """
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
            raise ValueError(
                "Invalid mode. Supported modes are 'b64_json' and 's3_json'."
            )


@batch
def _infer_fn(
    prompt: np.ndarray,
    negative_prompt: np.ndarray,
    num_images: np.ndarray,
    num_inference_steps: np.ndarray,
    guidance_scale: np.ndarray,
    mode: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Batch inference function for processing multiple requests.

    This function is decorated with @batch to handle multiple inference requests simultaneously.

    Args:
        prompt (np.ndarray): Array of prompts for image generation.
        negative_prompt (np.ndarray): Array of negative prompts.
        num_images (np.ndarray): Array of number of images to generate per prompt.
        num_inference_steps (np.ndarray): Array of inference steps for each request.
        guidance_scale (np.ndarray): Array of guidance scales.
        mode (np.ndarray): Array of output modes.

    Returns:
        Dict[str, np.ndarray]: A dictionary with the 'output' key containing the batch results.
    """
    prompts = [np.char.decode(p.astype("bytes"), "utf-8").item() for p in prompt]
    negative_prompts = [
        np.char.decode(p.astype("bytes"), "utf-8").item() for p in negative_prompt
    ]
    modes = [np.char.decode(m.astype("bytes"), "utf-8").item() for m in mode]

    LOGGER.debug(f"Prompts: {prompts}")
    LOGGER.debug(f"Negative Prompts: {negative_prompts}")
    LOGGER.debug(f"Num Images: {num_images}")
    LOGGER.debug(f"Num Inference Steps: {num_inference_steps}")
    LOGGER.debug(f"Guidance Scale: {guidance_scale}")
    LOGGER.debug(f"Modes: {modes}")

    outputs = []
    for idx, (prompt, neg_prompt, num_img, steps, scale, mode) in enumerate(
        zip(
            prompts,
            negative_prompts,
            num_images,
            num_inference_steps,
            guidance_scale,
            modes,
        )
    ):
        inference = SDXLLoraInference(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_images=int(num_img),
            num_inference_steps=int(steps),
            guidance_scale=float(scale),
            mode=mode,
        )
        result = inference.run_inference()
        json_result = json.dumps(result).encode("utf-8")
        outputs.append([json_result])
        LOGGER.debug(
            f"Generated result for prompt `{prompt}` with size {len(json_result)}"
        )

    LOGGER.debug(f"Prepared batch response of size: {len(outputs)}")
    return {"output": np.array(outputs)}


def main():
    """
    Initialize and start the Triton Inference Server with the SDXL Lora model.

    This function sets up the Triton server, binds the inference function,
    and starts serving inference requests.
    """
    triton_config = TritonConfig(exit_on_error=True)

    with Triton(config=triton_config) as triton:
        LOGGER.info("Loading the SDXL Lora pipeline")
        triton.bind(
            model_name=settings.TRITON_MODEL_NAME,
            infer_func=_infer_fn,
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
                max_batch_size=settings.MAX_BATCH_SIZE,
                batcher=DynamicBatcher(
                    max_queue_delay_microseconds=settings.MAX_QUEUE_DELAY_MICROSECONDS,
                ),
            ),
            strict=True,
        )
        triton.serve()


if __name__ == "__main__":
    main()
