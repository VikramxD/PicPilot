import random
from typing import Tuple
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from diffusers import FluxInpaintPipeline
from torchao.quantization import autoquant
from scripts.api_utils import accelerator
from optimum.quanto import freeze, qfloat8, quantize


class FluxInpaintingInference:
    """
    A class to perform image inpainting using the FLUX model from Hugging Face's Diffusers library.

    Attributes:
        MAX_SEED (int): The maximum value for a random seed.
        DEVICE (str): The device to run the model on ('cuda' or 'cpu').
        IMAGE_SIZE (int): The maximum size for the input image dimensions.
    """

    MAX_SEED = np.iinfo(np.int32).max
    IMAGE_SIZE = 1024
    DEVICE = 'cuda'

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    ):
        """
        Initializes the FluxInpaintingInference class with a specified model and data type.

        Args:
            model_name (str): The name of the model to be loaded from Hugging Face Hub.
            torch_dtype: The data type to be used by PyTorch (e.g., torch.float16).
        """
        self.pipeline = FluxInpaintPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.pipeline.vae.enable_slicing()
        self.pipeline.vae.enable_tiling()
        self.pipeline.transformer.to(memory_format=torch.channels_last)
        self.pipeline.transformer = torch.compile(self.pipeline.transformer, mode="max-autotune", fullgraph=True)
        quantize(self.pipeline.transformer,weights=qfloat8)
        freeze(self.pipeline.transformer)
        self.pipeline.to(self.DEVICE)
        



    @staticmethod
    def calculate_new_dimensions(
        original_dimensions: Tuple[int, int], max_dimension: int = 1024
    ) -> Tuple[int, int]:
        """
        Calculates new image dimensions while maintaining aspect ratio and ensuring divisibility by 32.

        Args:
            original_dimensions (Tuple[int, int]): The original width and height of the image.
            max_dimension (int): The maximum dimension size.

        Returns:
            Tuple[int, int]: The new width and height.
        """
        width, height = original_dimensions

        # Calculate scaling factor
        scaling_factor = min(max_dimension / width, max_dimension / height, 1.0)

        # Calculate new dimensions and make them divisible by 32
        new_width = int((width * scaling_factor) // 32 * 32)
        new_height = int((height * scaling_factor) // 32 * 32)

        # Ensure minimum size of 32x32
        new_width = max(32, new_width)
        new_height = max(32, new_height)

        return new_width, new_height

    def generate_inpainting(
        self,
        input_image: Image.Image,
        mask_image: Image.Image,
        prompt: str,
        seed: int = None,
        randomize_seed: bool = False,
        strength: float = 0.8,
        num_inference_steps: int = 50,
    ) -> Image.Image:
        """
        Generates an inpainted image based on the provided inputs.

        Args:
            input_image (Image.Image): The original image to be inpainted.
            mask_image (Image.Image): The mask indicating areas to be inpainted (white areas are inpainted).
            prompt (str): Text prompt guiding the inpainting.
            seed (int, optional): Seed for random number generation. Defaults to None.
            randomize_seed (bool, optional): Whether to randomize the seed. Defaults to False.
            strength (float, optional): Strength of the inpainting effect (0.0 to 1.0). Defaults to 0.8.
            num_inference_steps (int, optional): Number of denoising steps. Defaults to 50.

        Returns:
            Image.Image: The resulting inpainted image.
        """
        if randomize_seed or seed is None:
            seed = random.randint(0, self.MAX_SEED)

        generator = torch.Generator(device=self.DEVICE).manual_seed(seed)

        # Resize images
        new_width, new_height = self.calculate_new_dimensions(input_image.size)
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        mask_image = mask_image.resize((new_width, new_height), Image.LANCZOS)

        # Run inference
        result = self.pipeline(
            prompt=prompt,
            image=input_image,
            mask_image=mask_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]

        return result
