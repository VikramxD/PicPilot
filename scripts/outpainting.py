import torch
from PIL import Image, ImageDraw
import numpy as np
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
from scripts.controlnet_union import ControlNetModel_Union
from scripts.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline




class Outpainter:
    """
    A class for performing outpainting operations using Stable Diffusion XL.
    
    This class handles the setup and execution of outpainting tasks, including
    model initialization, image preparation, and the actual outpainting process.
    """

    def __init__(self):
        """Initialize the Outpainter by setting up the required models."""
        self.setup_model()

    def setup_model(self):
        """
        Set up and configure the SDXL model with ControlNet and VAE components.
        
        Downloads necessary model files, initializes components, and configures
        the pipeline for inference.
        """
        config_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="config_promax.json",
        )
        config = ControlNetModel_Union.load_config(config_file)
        controlnet_model = ControlNetModel_Union.from_config(config)
        model_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="diffusion_pytorch_model_promax.safetensors",
        )
        state_dict = load_state_dict(model_file)
        model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
            controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
        )
        model.to(device="cuda", dtype=torch.float16)

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        ).to("cuda")

        self.pipe = StableDiffusionXLFillPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0_Lightning",
            torch_dtype=torch.float16,
            vae=vae,
            controlnet=model,
            variant="fp16",
        ).to("cuda")

        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)

    def calculate_margins(self, target_size: tuple, new_width: int, new_height: int, alignment: str) -> tuple:
        """
        Calculate image margins based on alignment and dimensions.
        
        Args:
            target_size: Tuple of (width, height) for the target canvas size
            new_width: Width of the resized image
            new_height: Height of the resized image
            alignment: Position alignment ("Middle", "Left", "Right", "Top", "Bottom")
            
        Returns:
            tuple: (margin_x, margin_y) coordinates for image placement
        """
        if alignment == "Middle":
            margin_x = (target_size[0] - new_width) // 2
            margin_y = (target_size[1] - new_height) // 2
        elif alignment == "Left":
            margin_x = 0
            margin_y = (target_size[1] - new_height) // 2
        elif alignment == "Right":
            margin_x = target_size[0] - new_width
            margin_y = (target_size[1] - new_height) // 2
        elif alignment == "Top":
            margin_x = (target_size[0] - new_width) // 2
            margin_y = 0
        elif alignment == "Bottom":
            margin_x = (target_size[0] - new_width) // 2
            margin_y = target_size[1] - new_height
        else:
            margin_x = (target_size[0] - new_width) // 2
            margin_y = (target_size[1] - new_height) // 2

        margin_x = max(0, min(margin_x, target_size[0] - new_width))
        margin_y = max(0, min(margin_y, target_size[1] - new_height))
        
        return margin_x, margin_y

    def prepare_image_and_mask(self, image: Image, width: int, height: int, 
                             overlap_percentage: int, resize_option: str, 
                             custom_resize_percentage: int, alignment: str, 
                             overlap_left: bool, overlap_right: bool, 
                             overlap_top: bool, overlap_bottom: bool) -> tuple:
        """
        Prepare the input image and generate a mask for outpainting.
        
        Args:
            image: Input PIL Image
            width: Target width for output
            height: Target height for output
            overlap_percentage: Percentage of overlap for mask
            resize_option: Image resize option ("Full", "50%", "33%", "25%", "Custom")
            custom_resize_percentage: Custom resize percentage if resize_option is "Custom"
            alignment: Image alignment in the canvas
            overlap_left: Apply overlap on left side
            overlap_right: Apply overlap on right side
            overlap_top: Apply overlap on top side
            overlap_bottom: Apply overlap on bottom side
            
        Returns:
            tuple: (background_image, mask_image) prepared for outpainting
        """
        target_size = (width, height)
        scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        
        source = image.resize((new_width, new_height), Image.LANCZOS)

        resize_percentage = {
            "Full": 100,
            "50%": 50,
            "33%": 33,
            "25%": 25
        }.get(resize_option, custom_resize_percentage)

        resize_factor = resize_percentage / 100
        new_width = max(int(source.width * resize_factor), 64)
        new_height = max(int(source.height * resize_factor), 64)

        source = source.resize((new_width, new_height), Image.LANCZOS)

        overlap_x = max(int(new_width * (overlap_percentage / 100)), 1)
        overlap_y = max(int(new_height * (overlap_percentage / 100)), 1)

        margin_x, margin_y = self.calculate_margins(target_size, new_width, new_height, alignment)

        background = Image.new('RGB', target_size, (255, 255, 255))
        background.paste(source, (margin_x, margin_y))

        mask = Image.new('L', target_size, 255)
        white_gaps_patch = 2

        left_overlap = margin_x + (overlap_x if overlap_left else white_gaps_patch)
        right_overlap = margin_x + new_width - (overlap_x if overlap_right else white_gaps_patch)
        top_overlap = margin_y + (overlap_y if overlap_top else white_gaps_patch)
        bottom_overlap = margin_y + new_height - (overlap_y if overlap_bottom else white_gaps_patch)

        if alignment == "Left":
            left_overlap = margin_x + (overlap_x if overlap_left else 0)
        elif alignment == "Right":
            right_overlap = margin_x + new_width - (overlap_x if overlap_right else 0)
        elif alignment == "Top":
            top_overlap = margin_y + (overlap_y if overlap_top else 0)
        elif alignment == "Bottom":
            bottom_overlap = margin_y + new_height - (overlap_y if overlap_bottom else 0)

        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([left_overlap, top_overlap, right_overlap, bottom_overlap], fill=0)

        return background, mask

    def outpaint(self, image: Image, width: int, height: int, 
                overlap_percentage: int, num_inference_steps: int, 
                resize_option: str, custom_resize_percentage: int, 
                prompt_input: str, alignment: str,
                overlap_left: bool, overlap_right: bool, 
                overlap_top: bool, overlap_bottom: bool) -> Image:
        """
        Perform outpainting on the input image.
        
        Args:
            image: Input PIL Image to outpaint
            width: Target width for output
            height: Target height for output
            overlap_percentage: Percentage of overlap for mask
            num_inference_steps: Number of denoising steps
            resize_option: Image resize option
            custom_resize_percentage: Custom resize percentage
            prompt_input: Text prompt for generation
            alignment: Image alignment in canvas
            overlap_left: Apply overlap on left
            overlap_right: Apply overlap on right
            overlap_top: Apply overlap on top
            overlap_bottom: Apply overlap on bottom
            
        Returns:
            PIL.Image: Outpainted image
        """
        background, mask = self.prepare_image_and_mask(
            image, width, height, overlap_percentage, resize_option, 
            custom_resize_percentage, alignment, overlap_left, overlap_right, 
            overlap_top, overlap_bottom
        )
        
        cnet_image = background.copy()
        cnet_image.paste(0, (0, 0), mask)

        final_prompt = f"{prompt_input}, high quality, 4k"

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(final_prompt, "cuda", True)

        generator = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
            num_inference_steps=num_inference_steps
        )

        for output in generator:
            final_image = output

        final_image = final_image.convert("RGBA")
        cnet_image.paste(final_image, (0, 0), mask)

        return cnet_image