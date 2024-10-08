import torch
from PIL import Image, ImageDraw
import numpy as np
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

class Outpainter:
    def __init__(self):
        self.setup_model()

    def setup_model(self):
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

    def prepare_image_and_mask(self, image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
        target_size = (width, height)

        # Calculate the scaling factor to fit the image within the target size
        scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        
        # Resize the source image to fit within target size
        source = image.resize((new_width, new_height), Image.LANCZOS)

        # Apply resize option
        if resize_option == "Full":
            resize_percentage = 100
        elif resize_option == "50%":
            resize_percentage = 50
        elif resize_option == "33%":
            resize_percentage = 33
        elif resize_option == "25%":
            resize_percentage = 25
        else:  # Custom
            resize_percentage = custom_resize_percentage

        # Calculate new dimensions based on percentage
        resize_factor = resize_percentage / 100
        new_width = max(int(source.width * resize_factor), 64)
        new_height = max(int(source.height * resize_factor), 64)

        # Resize the image
        source = source.resize((new_width, new_height), Image.LANCZOS)

        # Calculate the overlap in pixels
        overlap_x = max(int(new_width * (overlap_percentage / 100)), 1)
        overlap_y = max(int(new_height * (overlap_percentage / 100)), 1)

        # Calculate margins based on alignment
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

        # Adjust margins to eliminate gaps
        margin_x = max(0, min(margin_x, target_size[0] - new_width))
        margin_y = max(0, min(margin_y, target_size[1] - new_height))

        # Create a new background image and paste the resized source image
        background = Image.new('RGB', target_size, (255, 255, 255))
        background.paste(source, (margin_x, margin_y))

        # Create the mask
        mask = Image.new('L', target_size, 255)
        white_gaps_patch = 2

        left_overlap = margin_x + (overlap_x if overlap_left else white_gaps_patch)
        right_overlap = margin_x + new_width - (overlap_x if overlap_right else white_gaps_patch)
        top_overlap = margin_y + (overlap_y if overlap_top else white_gaps_patch)
        bottom_overlap = margin_y + new_height - (overlap_y if overlap_bottom else white_gaps_patch)

        # Adjust overlaps for edge alignments
        if alignment == "Left":
            left_overlap = margin_x + (overlap_x if overlap_left else 0)
        elif alignment == "Right":
            right_overlap = margin_x + new_width - (overlap_x if overlap_right else 0)
        elif alignment == "Top":
            top_overlap = margin_y + (overlap_y if overlap_top else 0)
        elif alignment == "Bottom":
            bottom_overlap = margin_y + new_height - (overlap_y if overlap_bottom else 0)

        # Draw the mask
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle([left_overlap, top_overlap, right_overlap, bottom_overlap], fill=0)

        return background, mask

    def outpaint(self, image, width, height, overlap_percentage, num_inference_steps, resize_option, custom_resize_percentage, prompt_input, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
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

        # Iterate through the generator to get the final image
        for output in generator:
            final_image = output

        final_image = final_image.convert("RGBA")
        cnet_image.paste(final_image, (0, 0), mask)

        return cnet_image

# Usage example
if __name__ == "__main__":
    outpainter = Outpainter()
    
    # Load an example image
    image = Image.open("/root/PicPilot/sample_data/example4.jpg").convert("RGBA")
    
    # Set parameters
    width, height = 1024, 1024
    overlap_percentage = 10
    num_inference_steps = 8
    resize_option = "Full"
    custom_resize_percentage = 100
    prompt_input = "A Office"
    alignment = "Left"
    overlap_left = overlap_right = overlap_top = overlap_bottom = True
    
    # Run outpainting
    result = outpainter.outpaint(
        image, width, height, overlap_percentage, num_inference_steps,
        resize_option, custom_resize_percentage, prompt_input, alignment,
        overlap_left, overlap_right, overlap_top, overlap_bottom
    )
    
    # Save the result
    result.save("outpainted_image.png")