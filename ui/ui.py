import gradio as gr
import numpy as np
import requests
from io import BytesIO
import json
from PIL import Image
from pydantic import BaseModel, Field
from diffusers.utils import load_image

# API endpoints
sdxl_inference_endpoint = 'https://vikramsingh178-picpilot-server.hf.space/api/v1/product-diffusion/sdxl_v0_lora_inference'
kandinsky_inpainting_inference = 'https://vikramsingh178-picpilot-server.hf.space/api/v1/product-diffusion/inpainting'

class InputRequestSDXL(BaseModel):
    prompt: str
    num_inference_steps: int
    guidance_scale: float
    negative_prompt: str
    num_images: int
    mode: str

class InpaintingRequestKandinsky(BaseModel):
    prompt: str = Field(..., description="Prompt text for inference")
    negative_prompt: str = Field(..., description="Negative prompt text for inference")
    num_inference_steps: int = Field(..., description="Number of inference steps")
    strength: float = Field(..., description="Strength of the inference")
    guidance_scale: float = Field(..., description="Guidance scale for inference")
    mode: str = Field(..., description="Mode for output ('b64_json' or 's3_json')")
    num_images: int = Field(..., description="Number of images to generate")

def generate_sdxl_lora_image(prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, mode):
    payload = InputRequestSDXL(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images=num_images,
        mode=mode
    ).dict()
    
    try:
        response = requests.post(sdxl_inference_endpoint, json=payload)
        response.raise_for_status()
        response_data = response.json()
        url = response_data['url']
        image = load_image(url)
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error in SDXL-Lora API request: {e}")
        return None

def process_masked_image(img):
    if img is None or "composite" not in img:
        return None, None
    
    base_image = Image.fromarray(img["composite"]).convert("RGB")
    
    if "layers" in img and len(img["layers"]) > 0:
        alpha_channel = img["layers"][0][:, :, 3]
        mask = np.where(alpha_channel == 0, 0, 255).astype(np.uint8)
        mask = Image.fromarray(mask).convert("L")
    else:
        mask = Image.new("L", base_image.size, 0)
    
    return base_image, mask

def generate_outpainting(prompt, negative_prompt, num_inference_steps, strength, guidance_scale, mode, num_images, masked_image):
    base_image, mask = process_masked_image(masked_image)
    
    if base_image is None or mask is None:
        return None, None
    
    # Convert the images to bytes
    img_byte_arr = BytesIO()
    base_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    mask_byte_arr = BytesIO()
    mask.save(mask_byte_arr, format='PNG')
    mask_byte_arr = mask_byte_arr.getvalue()
    
    # Prepare the files for multipart/form-data
    files = {
        'image': ('image.png', img_byte_arr, 'image/png'),
        'mask_image': ('mask.png', mask_byte_arr, 'image/png'),
    }

    # Prepare the request data
    request_data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "mode": mode,
        "num_images": num_images
    }

    # Convert request_data to a JSON string
    request_data_json = json.dumps(request_data)

    # Prepare the form data
    form_data = {
        'request_data': request_data_json
    }
    
    try:
        response = requests.post(kandinsky_inpainting_inference, files=files, data=form_data)
        response.raise_for_status()
        response_data = response.json()
        url = response_data['url']
        outpainted_image = load_image(url)
        return mask, outpainted_image
    except requests.exceptions.RequestException as e:
        print(f"Error in Kandinsky Inpainting API request: {e}")
        return None, None

def generate_mask_preview(img):
    base_image, mask = process_masked_image(img)
    if mask is None:
        return None
    return mask

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
                image_preview = gr.Image(label="Generated Image (SDXL-Lora)", show_download_button=True, show_share_button=True, container=True)
                generate_button.click(generate_sdxl_lora_image, inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, mode], outputs=[image_preview])
                

    with gr.Tab("Inpainting"):
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    masked_image = gr.ImageMask(label="Upload Image and Draw Mask", format='png')
                    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
                    negative_prompt= gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here")
                    num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Inference Steps")
                    strength = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.8, label="Strength")
                    guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=7.5, label="Guidance Scale")
                    num_images= gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Number of Images")
                    mode_kandinsky = gr.Dropdown(choices=["s3_json", "b64_json"], value="s3_json", label="Mode")
                    generate_button = gr.Button("Generate Inpainting", variant='primary')
                    generate_mask_button_painting = gr.Button("Generate Mask", variant='secondary')

            with gr.Column(scale=1):
                mask_preview= gr.Image(label="Mask Preview", show_download_button=True, container=True)
                outpainted_image_preview = gr.Image(label="Outpainted Image (Kandinsky)", show_download_button=True, show_share_button=True, container=True)
                generate_mask_button_painting.click(generate_mask_preview, inputs=masked_image, outputs=[mask_preview])
                generate_button.click(generate_outpainting, 
                                                inputs=[prompt, negative_prompt, num_inference_steps, strength, guidance_scale, mode_kandinsky, num_images, masked_image], 
                                                outputs=[mask_preview, outpainted_image_preview])

demo.launch()