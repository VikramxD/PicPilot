from diffusers import DiffusionPipeline
import torch


class SDXLLoraInference:
    """
    Class for running inference using the SDXL-LoRA model to generate stunning product photographs.
    
    Args:
        prompt (str): The input prompt for generating the product photograph.
        num_inference_steps (int): The number of inference steps to perform.
        guidance_scale (float): The scale factor for guidance during inference.
    """
    def __init__(self, prompt: str, num_inference_steps: int, guidance_scale: float) -> None:
        self.model_path = "VikramSingh178/sdxl-lora-finetune-product-caption"
        self.pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        self.pipe.to("cuda")
        self.pipe.load_lora_weights(self.model_path)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.prompt = prompt
    

    def run_inference(self):
        """
        Runs inference using the SDXL-LoRA model to generate a stunning product photograph.
        
        Returns:
            images: The generated product photograph(s).
        """
        
        prompt = self.prompt
        images = self.pipe(prompt, num_inference_steps=self.num_inference_steps, guidance_scale=self.guidance_scale).images
        return images

inference = SDXLLoraInference(num_inference_steps=100, guidance_scale=2.5)
inference.run_inference(prompt= "A stunning 4k Shot of a Balenciaga X Anime Hoodie with a person wearing it in a party" )
