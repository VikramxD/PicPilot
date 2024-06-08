import gradio as gr
import requests
from pydantic import BaseModel

# Define your API endpoint
SDXL_LORA_API_URL = 'http://127.0.0.1:8000/api/v1/product-diffusion/sdxl_v0_lora_inference'

# Define the InpaintingRequest model
class InpaintingRequest(BaseModel):
    prompt: str
    num_inference_steps: int
    guidance_scale: float
    negative_prompt: str
    num_images: int
    mode: str

def generate_sdxl_lora_image(prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, mode):
    # Prepare the payload for SDXL LORA API
    payload = InpaintingRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images=num_images,
        mode=mode
    ).dict()
    
    response = requests.post(SDXL_LORA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get('image')
    else:
        return f"Error: {response.json().get('detail', 'Unknown error')}"

with gr.Blocks() as demo:
    with gr.Tab("SDXL LORA Inpainting"):
        gr.Markdown("## SDXL LORA Inpainting")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Parameters")
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here")
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt here")
                num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Inference Steps")
                guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.1, value=7.5, label="Guidance Scale")
                num_images = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Number of Images")
                mode = gr.Dropdown(choices=["s3_json", "default"], value="s3_json", label="Mode")
                generate_button = gr.Button(value="Generate Image")
            with gr.Column():
                gr.Markdown("### Output")
                output_image = gr.Image(label="Generated Image")
        generate_button.click(fn=generate_sdxl_lora_image, inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, mode], outputs=output_image)

demo.launch()
