import gradio as gr
import numpy as np
import requests
import json
from PIL import Image
from diffusers.utils import load_image
from io import BytesIO
from vars import base_url

# API endpoints
sdxl_inference_endpoint = f'{base_url}/api/v1/product-diffusion/sdxl_v0_lora_inference'
kandinsky_inpainting_inference = f'{base_url}/api/v1/product-diffusion/inpainting'

def generate_sdxl_lora_image(prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, mode):
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images": num_images,
        "mode": mode
    }
    
    response = requests.post(sdxl_inference_endpoint, json=payload)
    response.raise_for_status()
    response_data = response.json()
    url = response_data['url']
    image = load_image(url)
    return image

def generate_outpainting(prompt, negative_prompt, num_inference_steps, strength, guidance_scale, mode, num_images, image, width, height):
    # Convert the image to bytes
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Prepare the files for multipart/form-data
    files = {
        'image': ('image.png', img_byte_arr, 'image/png')
    }

    # Prepare the request data
    request_data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "strength": strength,
        "guidance_scale": guidance_scale,
        "mode": mode,
        "num_images": num_images,
        "width": width,
        "height": height
    }

    # Convert request_data to a JSON string
    request_data_json = json.dumps(request_data)

    # Prepare the form data
    form_data = {
        'request_data': request_data_json
    }
    
    response = requests.post(kandinsky_inpainting_inference, files=files, data=form_data)
    response.raise_for_status()
    response_data = response.json()
    image_url = response_data['image_url']
    mask_url = response_data['mask_url']
    outpainted_image = load_image(image_url)
    mask_image = load_image(mask_url)
    return outpainted_image, mask_image

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
                    input_image = gr.Image(label="Upload Image", type="pil")
                    prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
                    negative_prompt= gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here")
                    num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Inference Steps")
                    strength = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.8, label="Strength")
                    guidance_scale = gr.Slider(minimum=1.0, maximum=10.0, step=0.1, value=7.5, label="Guidance Scale")
                    num_images = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Number of Images")
                    mode_kandinsky = gr.Dropdown(choices=["s3_json", "b64_json"], value="s3_json", label="Mode")
                    width_slider = gr.Slider(minimum=512, maximum=1024, step=1, value=800, label="Image Width")
                    height_slider = gr.Slider(minimum=512, maximum=1024, step=1, value=800, label="Image Height")
                    generate_button = gr.Button("Generate Inpainting", variant='primary')

            with gr.Column(scale=1):
                outpainted_image_preview = gr.Image(label="Outpainted Image (Kandinsky)", show_download_button=True, show_share_button=True, container=True)
                mask_image_preview = gr.Image(label="Generated Mask", show_download_button=True, show_share_button=True, container=True)
                generate_button.click(generate_outpainting, inputs=[prompt, negative_prompt, num_inference_steps, strength, guidance_scale, mode_kandinsky, num_images, input_image, width_slider, height_slider], outputs=[outpainted_image_preview, mask_image_preview])

demo.launch()
