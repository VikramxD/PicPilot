"""
This module provides a class for generating videos from images using the CogVideoX model.
"""

import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image


class ImageToVideoPipeline:
    """
    A class to generate videos from images using the CogVideoX model.

    This class encapsulates the functionality of the CogVideoXImageToVideoPipeline,
    providing methods to generate video frames from an input image and save them as a video file.

    Attributes:
        pipe (CogVideoXImageToVideoPipeline): The underlying CogVideoX pipeline.
    """

    def __init__(
        self,
        model_path: str = "THUDM/CogVideoX-5b-I2V",
        device: str = "cuda:2",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize the ImageToVideoPipeline.

        Args:
            model_path (str): Path to the pretrained CogVideoX model.
            device (str): The device to run the model on (e.g., "cuda:2", "cpu").
            torch_dtype (torch.dtype): The torch data type to use for computations.
        """
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype
        )
        self.pipe.to(device)

    def generate(
        self,
        prompt: str,
        image: str | torch.Tensor,
        negative_prompt: str | None = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = True,
        height: int = 480,
        width: int = 720,
        num_videos_per_prompt: int = 1
    ) -> list:
        """
        Generate video frames from an input image.

        Args:
            prompt (str): The text prompt to guide the video generation.
            image (str | torch.Tensor): The input image path or tensor.
            negative_prompt (str | None): The negative prompt to guide the generation.
            num_frames (int): The number of frames to generate.
            num_inference_steps (int): The number of denoising steps.
            guidance_scale (float): The scale for classifier-free guidance.
            use_dynamic_cfg (bool): Whether to use dynamic CFG.
            height (int): The height of the output video frames.
            width (int): The width of the output video frames.
            num_videos_per_prompt (int): The number of videos to generate per prompt.

        Returns:
            list: A list of generated video frames.
        """
        if isinstance(image, str):
            image = load_image(image)

        result = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            height=height,
            width=width,
            num_videos_per_prompt=num_videos_per_prompt
        )
        return result.frames[0]

    
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = ImageToVideoPipeline(device="cuda:2")
    prompt = ("An astronaut hatching from an egg, on the surface of the moon the darkness and depth of space realised in the background.  ,High quality, ultrarealistic detail and breath-taking movie-like camera shot.")
    image_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    )

    # Generate and save the video
    pipeline.generate(
        prompt=prompt,
        image=image_url,
        output_file="custom_output.mp4",
        num_frames=60,
        num_inference_steps=75,
        guidance_scale=7.5,
        use_dynamic_cfg=True,
        height=640,
        width=960,
        fps=30
    )