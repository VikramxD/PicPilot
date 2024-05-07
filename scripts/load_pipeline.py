from config import MODEL_NAME,ADAPTER_NAME
import torch
from diffusers import DiffusionPipeline
from wandb.integration.diffusers import autolog
from config import PROJECT_NAME
autolog(init=dict(project=PROJECT_NAME))


def load_pipeline(model_name, adapter_name):
        pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(
            "cuda"
        )
        pipe.load_lora_weights(adapter_name)
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
        pipe.vae.decode = torch.compile(
            pipe.vae.decode, mode="max-autotune", fullgraph=True
        )
        pipe.fuse_qkv_projections()

        return pipe
    
loaded_pipeline = load_pipeline(MODEL_NAME, ADAPTER_NAME)
images = loaded_pipeline('toaster', num_inference_steps=30).images[0]
