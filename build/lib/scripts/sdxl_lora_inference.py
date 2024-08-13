from wandb.integration.diffusers import autolog
from diffusers import DiffusionPipeline
import torch
from config import PROJECT_NAME
autolog(init=dict(project=PROJECT_NAME))

class SDXLLoraInference:
    """
    Class for running inference using the SDXL-LoRA model to generate stunning product photographs.
    
    Args:
        num_inference_steps (int): The number of inference steps to perform.
        guidance_scale (float): The scale factor for guidance during inference.
    """
    def __init__(self, num_inference_steps: int, guidance_scale: float) -> None:
        self.model_path = "VikramSingh178/sdxl-lora-finetune-product-caption"
        self.pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        self.pipe.to("cuda")
        self.pipe.load_lora_weights(self.model_path)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    def run_inference(self, prompt):
        """
        Runs inference using the SDXL-LoRA model to generate a stunning product photograph.
        
        Args:
            prompt: The input prompt for generating the product photograph.
        
        Returns:
            images: The generated product photograph(s).
        """
        
        prompt = prompt
        images = self.pipe(prompt, num_inference_steps=self.num_inference_steps, guidance_scale=self.guidance_scale).images
        return images

inference = SDXLLoraInference(num_inference_steps=100, guidance_scale=2.5)
inference.run_inference(prompt= "A stunning 4k Shot of a Balenciaga X Anime Hoodie with a person wearing it in a party" )
