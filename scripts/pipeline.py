from diffusers import ControlNetModel,StableDiffusionControlNetInpaintPipeline,AutoPipelineForInpainting,KandinskyV22ControlnetImg2ImgPipeline,KandinskyV22PriorEmb2EmbPipeline
from diffusers.utils import load_image
import torch
from PIL import Image
import numpy as np
import cv2
import torch





class PipelineFetcher:
    """
    A class that fetches different pipelines for image processing.

    Args:
        controlnet_adapter_model_name (str): The name of the controlnet adapter model.
        controlnet_base_model_name (str): The name of the controlnet base model.
        kandinsky_model_name (str): The name of the Kandinsky model.
        image (str): The image to be processed.

    """

    def __init__(self, controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image: str):
        self.controlnet_adapter_model_name = controlnet_adapter_model_name
        self.controlnet_base_model_name = controlnet_base_model_name
        self.kandinsky_model_name = kandinsky_model_name
        self.image = image

    def ControlNetInpaintPipeline(self):
        """
        Fetches the ControlNet inpainting pipeline.

        Returns:
            pipe (StableDiffusionControlNetInpaintPipeline): The ControlNet inpainting pipeline.

        """
        controlnet = ControlNetModel.from_pretrained(self.controlnet_adapter_model_name, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.controlnet_base_model_name, controlnet=controlnet, torch_dtype=torch.float16
        )
        pipe.to('cuda')

        return pipe

    def KandinskyPipeline(self):
        """
        Fetches the Kandinsky pipeline.

        Returns:
            pipe (AutoPipelineForInpainting): The Kandinsky pipeline.

        """
        pipe = AutoPipelineForInpainting.from_pretrained(self.kandinsky_model_name, torch_dtype=torch.float16)
        pipe.to('cuda')
        return pipe

    def KandinskyPriorPipeline(self):
        """
        Fetches the Kandinsky prior pipeline.

        Returns:
            prior_pipeline (KandinskyV22PriorEmb2EmbPipeline): The Kandinsky prior pipeline.

        """
        prior_pipeline = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16, use_safetensors=False
        ).to("cuda")
        return prior_pipeline

    def KandinskyImg2ImgPipeline(self):
        """
        Fetches the Kandinsky img2img pipeline.

        Returns:
            img2img_pipeline (KandinskyV22ControlnetImg2ImgPipeline): The Kandinsky img2img pipeline.

        """
        img2img_pipeline = KandinskyV22ControlnetImg2ImgPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-2-controlnet-depth", torch_dtype=torch.float16, use_safetensors=False
        ).to("cuda")
        return img2img_pipeline
    
   

def fetch_control_pipeline(controlnet_adapter_model_name,controlnet_base_model_name,kandinsky_model_name,image):
    pipe_fetcher = PipelineFetcher(controlnet_adapter_model_name,controlnet_base_model_name,kandinsky_model_name,image)
    pipe = pipe_fetcher.ControlNetInpaintPipeline()
    return pipe

def fetch_kandinsky_pipeline(controlnet_adapter_model_name,controlnet_base_model_name,kandinsky_model_name,image):
    pipe_fetcher = PipelineFetcher(controlnet_adapter_model_name,controlnet_base_model_name,kandinsky_model_name,image)
    pipe = pipe_fetcher.KandinskyPipeline()
    return pipe

def fetch_kandinsky_prior_pipeline(controlnet_adapter_model_name,controlnet_base_model_name,kandinsky_model_name,image):
    pipe_fetcher = PipelineFetcher(controlnet_adapter_model_name,controlnet_base_model_name,kandinsky_model_name,image)
    pipe = pipe_fetcher.KandinskyPriorPipeline()
    return pipe

def fetch_kandinsky_img2img_pipeline(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image):
    """
    Fetches the Kandinsky image-to-image pipeline.

    Args:
        controlnet_adapter_model_name (str): The name of the controlnet adapter model.
        controlnet_base_model_name (str): The name of the controlnet base model.
        kandinsky_model_name (str): The name of the Kandinsky model.
        image: The input image.

    Returns:
        pipe: The Kandinsky image-to-image pipeline.
    """
    pipe_fetcher = PipelineFetcher(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image)
    pipe = pipe_fetcher.KandinskyImg2ImgPipeline()
    return pipe
def fetch_kandinsky_img2img_pipeline(controlnet_adapter_model_name,controlnet_base_model_name,kandinsky_model_name,image):
    pipe_fetcher = PipelineFetcher(controlnet_adapter_model_name,controlnet_base_model_name,kandinsky_model_name,image)
    pipe = pipe_fetcher.KandinskyImg2ImgPipeline()
    return pipe
