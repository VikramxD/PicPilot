import requests
import torch
from controlnet_aux import ZoeDetector
from PIL import Image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLInpaintPipeline
)
from typing import Optional
from api_utils import ImageAugmentation
import lightning.pytorch as pl 
pl.seed_everything(42)






class OutpaintingProcessor:
    """
    A class for processing and outpainting images using Stable Diffusion XL.

    This class encapsulates the entire pipeline for loading an image,
    generating a depth map, creating a temporary background, and performing
    the final outpainting.
    """

    def __init__(self, target_size=(1024, 1024)):
        """
        Initialize the OutpaintingProcessor with necessary models and pipelines.

        Args:
            target_size (tuple): The target size for the output image (width, height).
        """
        self.target_size = target_size
        print("Initializing models and pipelines...")
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(self.device)
        self.zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
        self.controlnets = [
            ControlNetModel.from_pretrained("destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"),
            ControlNetModel.from_pretrained("diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch.float16,variant='fp16')
            ]

        print("Setting up initial pipeline...")
        self.controlnet_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", 
            controlnet=self.controlnets, vae=self.vae
        ).to(self.device)

        print("Setting up inpaint pipeline...")
        self.inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained("OzzyGT/RealVisXL_V4.0_inpainting",torch_dtype=torch.float16,
            variant="fp16",
            vae=self.vae,
        ).to(self.device)

        print("Initialization complete.")

    def load_and_preprocess_image(self, image_url):
        """
        Load an image from a URL and preprocess it for outpainting.

        Args:
            image_url (str): URL of the image to process.

        Returns:
            tuple: A tuple containing the resized original image and the background image.
        """
        original_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGBA")
        return self.scale_and_paste(original_image, self.target_size)

    def scale_and_paste(self, original_image, target_size, scale_factor=0.95):
        """
        Scale the original image and paste it onto a background of the target size.

        Args:
            original_image (PIL.Image): The original image to process.
            target_size (tuple): The target size (width, height) for the output image.
            scale_factor (float): Factor to scale down the image to leave some padding (default: 0.95).

        Returns:
            tuple: A tuple containing the resized original image and the background image.
        """
        target_width, target_height = target_size
        aspect_ratio = original_image.width / original_image.height

        if (target_width / target_height) < aspect_ratio: 
            new_width = int(target_width * scale_factor)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = int(target_height * scale_factor)
            new_width = int(new_height * aspect_ratio)

        resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
        background = Image.new("RGBA", target_size, "white")
        x = (target_width - new_width) // 2
        y = (target_height - new_height) // 2
        background.paste(resized_original, (x, y), resized_original)

        return resized_original, background

    def generate_depth_map(self, image):
        """
        Generate a depth map for the given image using the Zoe model.

        Args:
            image (PIL.Image): The image to generate a depth map for.

        Returns:
            PIL.Image: The generated depth map.
        """
        return self.zoe(image, detect_resolution=512, image_resolution=self.target_size[0])

    def generate_image(self, prompt, negative_prompt, inpaint_image, zoe_image,guidance_scale,num_inference_steps):
        """
        Generate an image using the initial  pipeline.

        Args:
            prompt (str): The prompt for image generation.
            negative_prompt (str): The negative prompt for image generation.
            inpaint_image (PIL.Image): The image to inpaint.
            zoe_image (PIL.Image): The depth map image.
            seed (int, optional): Seed for random number generation.

        Returns:
            PIL.Image: The generated image.
        """
       
        return self.initial_pipeline(
            prompt,
            negative_prompt=negative_prompt,
            image=[inpaint_image, zoe_image],
            guidance_scale=guidance_scale,
            num_inference_steps=25,
            controlnet_conditioning_scale=[0.5, 0.8],
            control_guidance_end=[0.9, 0.6],
        ).images[0]

    def create_mask(self, image, segmentation_model, detection_model):
        """
        Create a mask for the final outpainting process.

        Args:
            image (PIL.Image): The original image.
            segmentation_model (str): The segmentation model identifier.
            detection_model (str): The detection model identifier.

        Returns:
            PIL.Image: The created mask.
        """
        image_augmenter = ImageAugmentation(self.target_size[0], self.target_size[1])
        mask_image = image_augmenter.generate_mask_from_bbox(image, segmentation_model,detection_model)
        inverted_mask = image_augmenter.invert_mask(mask_image)
        return inverted_mask

    def generate_outpainting(self, prompt, negative_prompt, image, mask, seed:Optional[int]=42):
        """
        Generate the final outpainted image.

        Args:
            prompt (str): The prompt for image generation.
            negative_prompt (str): The negative prompt for image generation.
            image (PIL.Image): The image to outpaint.
            mask (PIL.Image): The mask for outpainting.
            seed (int, optional): Seed for random number generation.

        Returns:
            PIL.Image: The final outpainted image.
        """
        
        return self.inpaint_pipeline(
            prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            guidance_scale=10.0,
            strength=0.8,
            num_inference_steps=30,
        ).images[0]

    def process(self, image_url, initial_prompt, final_prompt, negative_prompt=""):
        """
        Process an image through the entire outpainting pipeline.

        Args:
            image_url (str): URL of the image to process.
            initial_prompt (str): Prompt for the initial image generation.
            final_prompt (str): Prompt for the final outpainting.
            negative_prompt (str, optional): Negative prompt for both stages.

        Returns:
            PIL.Image: The final outpainted image.
        """
        print("Loading and preprocessing image...")
        resized_img, background_image = self.load_and_preprocess_image(image_url)
        
        print("Generating depth map...")
        image_zoe = self.generate_depth_map(background_image)

        print("Generating initial image...")
        temp_image = self.generate_image(initial_prompt, negative_prompt, background_image, image_zoe)
        x = (self.target_size[0] - resized_img.width) // 2
        y = (self.target_size[1] - resized_img.height) // 2
        temp_image.paste(resized_img, (x, y), resized_img)
        print("Creating mask for outpainting...")
        final_mask = self.create_mask(temp_image, "facebook/sam-vit-large", "yolov8l")
        mask_blurred = self.inpaint_pipeline.mask_processor.blur(final_mask, blur_factor=20)
        print("Generating final outpainted image...")
        final_image = self.generate_outpainting(final_prompt, negative_prompt, temp_image, mask_blurred)
        final_image.paste(resized_img, (x, y), resized_img)
        return final_image

def main():
    processor = OutpaintingProcessor(target_size=(1024, 1024))  # Set to 720p resolution
    result = processor.process(
        "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/outpainting/BMW_i8_Safety_Car_Front.png?download=true",
        "a car on the highway",
        "high quality photo of a car on the highway, shadows, highly detailed")
    result.save("outpainted_result.png")
    print("Outpainting complete. Result saved as 'outpainted_result.png'")

if __name__ == "__main__":
    main()
