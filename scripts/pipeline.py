from diffusers import ControlNetModel,StableDiffusionControlNetInpaintPipeline,AutoPipelineForInpainting
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


def fetch_control_pipeline(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image):
    """
    Fetches the control pipeline for image processing.

    Args:
        controlnet_adapter_model_name (str): The name of the controlnet adapter model.
        controlnet_base_model_name (str): The name of the controlnet base model.
        kandinsky_model_name (str): The name of the Kandinsky model.
        image: The input image for processing.

    Returns:
        pipe: The control pipeline for image processing.
    """
    pipe_fetcher = PipelineFetcher(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image)
    pipe = pipe_fetcher.ControlNetInpaintPipeline()
    return pipe


def fetch_kandinsky_pipeline(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image):
    """
    Fetches the Kandinsky pipeline.

    Args:
        controlnet_adapter_model_name (str): The name of the controlnet adapter model.
        controlnet_base_model_name (str): The name of the controlnet base model.
        kandinsky_model_name (str): The name of the Kandinsky model.
        image: The input image.

    Returns:
        pipe: The Kandinsky pipeline.
    """
    pipe_fetcher = PipelineFetcher(controlnet_adapter_model_name, controlnet_base_model_name, kandinsky_model_name, image)
    pipe = pipe_fetcher.KandinskyPipeline()
    return pipe



