import gradio as gr
import requests
from pydantic import BaseModel, Field
from diffusers.utils import load_image
from io import BytesIO
import json
from PIL import Image

sdxl_inference_endpoint = 'https://vikramsingh178-picpilot-server.hf.space/api/v1/product-diffusion/sdxl_v0_lora_inference'
sdxl_batch_inference_endpoint = 'https://vikramsingh178-picpilot-server.hf.space/api/v1/product-diffusion/sdxl_v0_lora_inference/batch'
kandinsky_inpainting_inference = 'https://vikramsingh178-picpilot-server.hf.space/api/v1/product-diffusion/inpainting'

class InputRequest(BaseModel):
    prompt: str
    num_inference_steps: int
    guidance_scale: float
    negative_prompt: str
    num_images: int
    mode: str

class InpaintingRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text for inference")
    negative_prompt: str = Field(..., description="Negative prompt text for inference")
    num_inference_steps: int = Field(..., description="Number of inference steps")
    strength: float = Field(..., description="Strength of the inference")
    guidance_scale: float = Field(..., description="Guidance scale for inference")
    mode: str = Field(..., description="Mode for output ('b64_json' or 's3_json')")
    num_images: int = Field(..., description="Number of images to generate")

async def generate_sdxl_lora_image(prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, mode):
    payload = InputRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images=num_images,
        mode=mode
    ).dict()
    
    response = requests.post(sdxl_inference_endpoint, json=payload)
    response = response.json()
    url = response['url']
    image = load_image(url)
    return image

def process_masked_image(img):
    base_image = Image.fromarray(img['image'])
    mask = Image.fromarray(img['mask'])
    return base_image, mask

def generate_outpainting(prompt, negative_prompt, num_inference_steps, strength, guidance_scale, mode, num_images, masked_image):
    base_image, mask = process_masked_image(masked_image)
    
    # Convert the images to bytes
    img_byte_arr = BytesIO()
    base_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    mask_byte_arr = BytesIO()
    mask.save(mask_byte_arr, format='PNG')
    mask_byte_arr = mask_byte_arr.getvalue()
    
    # Prepare the payload for multipart/form-data
    files = {
        'image': ('image.png', img_byte_arr, 'image/png'),
        'mask_image': ('mask.png', mask_byte_arr, 'image/png'),
    }

    # Prepare the request data
    request_data = InpaintingRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=guidance_scale,
        mode=mode,
        num_images=num_images
    ).dict()

    # Add the JSON-encoded request data to the files dictionary
    files['request_data'] = ('request_data.json', json.dumps(request_data), 'application/json')
    
    response = requests.post(kandinsky_inpainting_inference, files=files)
    response.raise_for_status()
    response = response.json()
    url = response['url']
    image = load_image(url)
    return image

with gr.Blocks(theme='VikramSingh178/Webui-Theme') as demo:
    with gr.Tab("SdxL-Lora"):
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
                    negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here")
                    num_inference_steps = gr.Slider(minimum=1, maximum=1000, step=1, value=20, label="Inference Steps")
                    guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=7.5, label="Guidance Scale")
                    num_images = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Number of Images")
                    mode = gr.Dropdown(choices=["s3_json", "b64_json"], value="s3_json", label="Mode")
                    generate_button = gr.Button("Generate Image", variant='primary')
            
            with gr.Column(scale=1):
                image_preview = gr.Image(label="Generated Image", show_download_button=True, show_share_button=True, container=True)
                generate_button.click(generate_sdxl_lora_image, inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, mode], outputs=[image_preview])

    with gr.Tab("Inpainting"):
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    masked_image = gr.ImageMask(label="Upload Image and Draw Mask")
                    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
                    negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here")
                    num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Inference Steps")
                    strength = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.8, label="Strength")
                    guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=7.5, label="Guidance Scale")
                    num_images = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Number of Images")
                    mode = gr.Dropdown(choices=["s3_json", "b64_json"], value="s3_json", label="Mode")
                    generate_button = gr.Button("Generate Inpainting", variant='primary')

            with gr.Column(scale=1):
                image_preview = gr.Image(label="Inpainted Image", show_download_button=True, show_share_button=True, container=True)
                generate_button.click(generate_outpainting, inputs=[prompt, negative_prompt, num_inference_steps, strength, guidance_scale, mode, num_images, masked_image], outputs=[image_preview])

demo.launch()