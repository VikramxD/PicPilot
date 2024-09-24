import requests
import base64
import json

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def test_flux_inpainting(base_url, input_image_path, mask_image_path):
    payload = {
        "model_type": "flux_inpainting",
        "prompt": "A beautiful landscape",
        "input_image": encode_image(input_image_path),
        "mask_image": encode_image(mask_image_path),
        "strength": 0.8,
        "seed": 42,
        "num_inference_steps": 50
    }
    
    response = requests.post(f"{base_url}/predict", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("Flux Inpainting Test Successful")
        print(f"Result URL: {result.get('result_url')}")
        print(f"Prompt: {result.get('prompt')}")
        print(f"Seed: {result.get('seed')}")
    else:
        print(f"Flux Inpainting Test Failed. Status Code: {response.status_code}")
        print(f"Response: {response.text}")

def test_sdxl_lora(base_url):
    payload = {
        "model_type": "sdxl_lora",
        "prompt": "A futuristic cityscape",
        "negative_prompt": "ugly, blurry",
        "num_images": 3,
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "mode": "s3_json"
    }
    
    response = requests.post(f"{base_url}/predict", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print("SDXL Lora Test Successful")
        print(f"Result URL: {result.get('result_url')}")
    else:
        print(f"SDXL Lora Test Failed. Status Code: {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    base_url = "http://localhost:8000"  # Change this to your server's URL
    input_image_path = "/root/PicPilot/sample_data/image.jpg"  # Change this to your input image path
    mask_image_path = "/root/PicPilot/sample_data/mask.png"  # Change this to your mask image path
    
    print("Testing Flux Inpainting Model...")
    test_flux_inpainting(base_url, input_image_path, mask_image_path)
    
    print("\nTesting SDXL Lora Model...")
    test_sdxl_lora(base_url)