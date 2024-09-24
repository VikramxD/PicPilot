import requests
import json
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

url = "http://127.0.0.1:8000/predict"
input_image_path = "/root/PicPilot/sample_data/image.jpg"
mask_image_path = "/root/PicPilot/sample_data/mask.png"

# Prepare the JSON data
json_data = {
    "prompt": "4k Nike",
    "strength": 0.8,
    "seed": 42,
    "num_inference_steps": 20,
    "input_image": encode_image(input_image_path),
    "mask_image": encode_image(mask_image_path)
}

# Send the POST request
response = requests.post(url, json=json_data)

# Print the response
print(response.json())