import random
from typing import Tuple
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from diffusers import FluxInpaintPipeline, FluxTransformer2DModel
from torchao.ao.quantization import quantize_, int8_weight_only

class FluxInpaintingInference:
    """
    A class to perform image inpainting using the FLUX model with int8 quantization for efficient inference.

    Attributes:
        MAX_SEED (int): The maximum value for a random seed.
        DEVICE (str): The device to run the model on ('cuda' or 'cpu').
        IMAGE_SIZE (int): The maximum size for the input image dimensions.
    """

    MAX_SEED = np.iinfo(np.int32).max
    IMAGE_SIZE = 1024
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    _pipeline = None

    @classmethod
    @lru_cache(maxsize=1)
    def get_pipeline(cls, model_name: str, torch_dtype):
        """
        Loads and caches the FluxInpaintPipeline with int8 quantization.

        Args:
            model_name (str): The name of the model to be loaded from Hugging Face Hub.
            torch_dtype: The data type to be used by PyTorch.

        Returns:
            FluxInpaintPipeline: The loaded and optimized pipeline.
        """
        if cls._pipeline is None:
            # Load the transformer with int8 quantization
            transformer = FluxTransformer2DModel.from_pretrained(
                model_name, 
                subfolder="transformer", 
                torch_dtype=torch_dtype
            )
            quantize_(transformer, int8_weight_only())

            # Load the rest of the pipeline
            cls._pipeline = FluxInpaintPipeline.from_pretrained(
                model_name, 
                transformer=transformer, 
                torch_dtype=torch_dtype
            )

            # Additional optimizations
            cls._pipeline.vae.enable_slicing()
            cls._pipeline.vae.enable_tiling()
            cls._pipeline.to(cls.DEVICE)

        return cls._pipeline

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    ):
        """
        Initializes the FluxInpaintingInference class with a specified model and optimizations.

        Args:
            model_name (str): The name of the model to be loaded from Hugging Face Hub.
            torch_dtype: The data type to be used by PyTorch (e.g., torch.bfloat16).
        """
        self.pipeline = self.get_pipeline(model_name, torch_dtype)

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
        guidance_scale: float = 0.0,
        max_sequence_length: int = 256,
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
            guidance_scale (float, optional): Scale for classifier-free guidance. Defaults to 0.0.
            max_sequence_length (int, optional): Maximum sequence length for the transformer. Defaults to 256.

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
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
        ).images[0]

        return result