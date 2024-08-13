import torch
from controlnet_aux import ZoeDetector
from PIL import Image
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline, StableDiffusionXLInpaintPipeline
from scripts.api_utils import ImageAugmentation, accelerator
import lightning.pytorch as pl
from rembg import remove

pl.seed_everything(42)


class ControlNetZoeDepthOutpainting:
    """
    A class for processing and outpainting images using Stable Diffusion XL.

    This class encapsulates the entire pipeline for loading an image,
    generating a depth map, creating a temporary background, and performing
    the final outpainting.
    """

    def __init__(self, target_size: tuple[int, int] = (1024, 1024)):
        """
        Initialize the ImageOutpaintingProcessor with necessary models and pipelines.

        Args:
            target_size (tuple[int, int]): The target size for the output image (width, height).
        """
        self.target_size = target_size
        print("Initializing models and pipelines...")
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(accelerator())
        self.zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
        self.controlnets = [
            ControlNetModel.from_pretrained("destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"),
            ControlNetModel.from_pretrained("diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch.float16)
        ]
        print("Setting up sdxl pipeline...")
        self.controlnet_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained("SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", controlnet=self.controlnets, vae=self.vae).to(accelerator())
        print("Setting up inpaint pipeline...")
        self.inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained("OzzyGT/RealVisXL_V4.0_inpainting", torch_dtype=torch.float16, variant="fp16", vae=self.vae).to(accelerator())

    def load_and_preprocess_image(self, image_path: str) -> tuple[Image.Image, Image.Image]:
        """
        Load an image from a file path and preprocess it for outpainting.

        Args:
            image_path (str): Path of the image to process.

        Returns:
            tuple[Image.Image, Image.Image]: A tuple containing the resized original image and the background image.
        """
        original_image = Image.open(image_path).convert("RGBA")
        original_image = remove(original_image)
        return self.scale_and_paste(original_image, self.target_size)

    def scale_and_paste(self, original_image: Image.Image, target_size: tuple[int, int], scale_factor: float = 0.95) -> tuple[Image.Image, Image.Image]:
        """
        Scale the original image and paste it onto a background of the target size.

        Args:
            original_image (Image.Image): The original image to process.
            target_size (tuple[int, int]): The target size (width, height) for the output image.
            scale_factor (float): Factor to scale down the image to leave some padding (default: 0.95).

        Returns:
            tuple[Image.Image, Image.Image]: A tuple containing the resized original image and the background image.
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

    def generate_depth_map(self, image: Image.Image) -> Image.Image:
        """
        Generate a depth map for the given image using the Zoe model.

        Args:
            image (Image.Image): The image to generate a depth map for.

        Returns:
            Image.Image: The generated depth map.
        """
        return self.zoe(image, detect_resolution=512, image_resolution=self.target_size[0])

    def generate_base_image(self, prompt: str, negative_prompt: str, inpaint_image: Image.Image, zoe_image: Image.Image, guidance_scale: float, controlnet_num_inference_steps: int, controlnet_conditioning_scale: float, control_guidance_end: float) -> Image.Image:
        """
        Generate an image using the controlnet pipeline.

        Args:
            prompt (str): The prompt for image generation.
            negative_prompt (str): The negative prompt for image generation.
            inpaint_image (Image.Image): The image to inpaint.
            zoe_image (Image.Image): The depth map image.
            guidance_scale (float): Guidance scale for controlnet.
            controlnet_num_inference_steps (int): Number of inference steps for controlnet.
            controlnet_conditioning_scale (float): Conditioning scale for controlnet.
            control_guidance_end (float): Guidance end for controlnet.

        Returns:
            Image.Image: The generated image.
        """
        return self.controlnet_pipeline(
            prompt,
            negative_prompt=negative_prompt,
            image=[inpaint_image, zoe_image],
            guidance_scale=guidance_scale,
            num_inference_steps=controlnet_num_inference_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_end=control_guidance_end,
        ).images[0]

    def create_mask(self, image: Image.Image, segmentation_model: str, detection_model: str) -> Image.Image:
        """
        Create a mask for the final outpainting process.

        Args:
            image (Image.Image): The original image.
            segmentation_model (str): The segmentation model identifier.
            detection_model (str): The detection model identifier.

        Returns:
            Image.Image: The created mask.
        """
        image_augmenter = ImageAugmentation(self.target_size[0], self.target_size[1], roi_scale=0.4)
        mask_image = image_augmenter.generate_mask_from_bbox(image, segmentation_model, detection_model)
        inverted_mask = image_augmenter.invert_mask(mask_image)
        return inverted_mask

    def generate_outpainting(self, prompt: str, negative_prompt: str, image: Image.Image, mask: Image.Image, guidance_scale: float, strength: float, num_inference_steps: int) -> Image.Image:
        """
        Generate the final outpainted image.

        Args:
            prompt (str): The prompt for image generation.
            negative_prompt (str): The negative prompt for image generation.
            image (Image.Image): The image to outpaint.
            mask (Image.Image): The mask for outpainting.
            guidance_scale (float): Guidance scale for inpainting.
            strength (float): Strength for inpainting.
            num_inference_steps (int): Number of inference steps for inpainting.

        Returns:
            Image.Image: The final outpainted image.
        """
        return self.inpaint_pipeline(
            prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
        ).images[0]

    def run_pipeline(self, image_path: str, controlnet_prompt: str, controlnet_negative_prompt: str, controlnet_conditioning_scale: float, controlnet_guidance_scale: float, controlnet_num_inference_steps: int, controlnet_guidance_end: float, inpainting_prompt: str, inpainting_negative_prompt: str, inpainting_guidance_scale: float, inpainting_strength: float, inpainting_num_inference_steps: int) -> Image.Image:
        """
        Process an image through the entire outpainting pipeline.

        Args:
            image_path (str): Path of the image to process.
            controlnet_prompt (str): Prompt for the controlnet image generation.
            controlnet_negative_prompt (str): Negative prompt for controlnet image generation.
            controlnet_conditioning_scale (float): Conditioning scale for controlnet.
            controlnet_guidance_scale (float): Guidance scale for controlnet.
            controlnet_num_inference_steps (int): Number of inference steps for controlnet.
            controlnet_guidance_end (float): Guidance end for controlnet.
            inpainting_prompt (str): Prompt for the inpainting image generation.
            inpainting_negative_prompt (str): Negative prompt for inpainting image generation.
            inpainting_guidance_scale (float): Guidance scale for inpainting.
            inpainting_strength (float): Strength for inpainting.
            inpainting_num_inference_steps (int): Number of inference steps for inpainting.

        Returns:
            Image.Image: The final outpainted image.
        """
        print("Loading and preprocessing image")
        resized_img, background_image = self.load_and_preprocess_image(image_path)
        print("Generating depth map")
        image_zoe = self.generate_depth_map(background_image)
        print("Generating initial image")
        temp_image = self.generate_base_image(controlnet_prompt, controlnet_negative_prompt, background_image, image_zoe,
                                              controlnet_guidance_scale, controlnet_num_inference_steps, controlnet_conditioning_scale, controlnet_guidance_end)
        x = (self.target_size[0] - resized_img.width) // 2
        y = (self.target_size[1] - resized_img.height) // 2
        temp_image.paste(resized_img, (x, y), resized_img)
        print("Creating mask for outpainting")
        final_mask = self.create_mask(temp_image, "facebook/sam-vit-large", "yolov8l")
        mask_blurred = self.inpaint_pipeline.mask_processor.blur(final_mask, blur_factor=20)
        print("Generating final outpainted image")
        final_image = self.generate_outpainting(inpainting_prompt, inpainting_negative_prompt, temp_image, mask_blurred,
                                                inpainting_guidance_scale, inpainting_strength, inpainting_num_inference_steps)
        final_image.paste(resized_img, (x, y), resized_img)
        return final_image


def main():
    processor = ControlNetZoeDepthOutpainting(target_size=(1024, 1024))
    result = processor.run_pipeline("/home/PicPilot/sample_data/example1.jpg",
                                    "product in the kitchen",
                                    "low resolution, Bad Resolution",
                                     0.9,
                                     7.5,
                                     50,
                                     0.6,
                                     "Editorial Photography of the Pot in the kitchen",
                                     "low Resolution, Bad Resolution",
                                     8,
                                     0.7,
                                     30)
    result.save("outpainted_result.png")
    print("Outpainting complete. Result saved as 'outpainted_result.png'")


if __name__ == "__main__":
    main()
