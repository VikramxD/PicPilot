
# 🚀 PicPilot 

[![GitHub Stars](https://img.shields.io/github/stars/VikramxD/Picpilot?style=social)](https://github.com/YourGitHubUsername/picpilot/stargazers)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/VikramxD/Picpilot/blob/main/LICENSE)


> PicPilot is a scalable solution that leverages state-of-the-art Text to Image Models to extend and enhance images and create product photography in seconds for your brand. Whether you're working with existing product images or need to generate new visuals, PicPilot prepares your visual content to be stunning, professional, and ready for marketing applications.



## Features
✅ Flux Inpainting for detailed image editing  
✅ SDXL (Stable Diffusion XL) with LoRA for high-quality image generation  
✅ SDXL Outpainting for extending images seamlessly  
✅ Image to Video Generation using CogvideoX   
✅ Batch processing support with configurable batch sizes and timeouts  

### Why PicPilot?
Creating professional product photography and visual narratives can be time-consuming and expensive. PicPilot aims to revolutionize this process by offering an AI-powered platform where you can enhance existing images, generate new ones, or even convert images to videos, creating stunning visuals for your brand in seconds.

## Installation

```bash
git clone https://github.com/VikramxD/Picpilot
cd Picpilot
```



Install Dependencies:

```bash
./run.sh
```

### 🛳️ Docker

To use PicPilot with Docker, execute the following commands:

```bash
docker pull vikram1202/picpilot:latest
docker run --gpus all -p 8000:8000 vikram1202/picpilot:latest
```

Alternatively, if you prefer to build the Docker image locally:

```bash
docker build -t picpilot .
docker run --gpus all -p 8000:8000 picpilot
```

## Usage

Run the Server:

```bash
cd api
python picpilot.py
```

This will start the server on port 8000 with all available API endpoints.

## API Endpoints

PicPilot offers the following API endpoints:

| Endpoint | Path | Purpose | Max Batch Size | Batch Timeout |
|----------|------|---------|----------------|---------------|
| Flux Inpainting | `/api/v2/painting/flux` | Detailed image editing and inpainting | 4 | 0.1 seconds |
| SDXL Generation | `/api/v2/generate/sdxl` | High-quality image generation using SDXL with LoRA | Configured in `tti_settings` | Configured in `tti_settings` |
| SDXL Outpainting | `/api/v2/painting/sdxl_outpainting` | Extending images seamlessly | 4 | 0.1 seconds |
| Image to Video | `/api/v2/image2video/cogvideox` | Converting images to videos | 1 | 0.1 seconds |

## Next Features
-  Support for Image Editing in FLUX Models
-  Support for Custom Flux LORA'S 
-  Support for CogvideoX finetuning

## Limitations
- Requires Powerful GPU's to Run  for optimal performance Especially the FLUX Models 
- Processing time may vary depending on the complexity of the task and input size
- Image to video conversion is limited to one image at a time

## License
PicPilot is licensed under the MIT license. See `LICENSE` for more information.

## Acknowledgements

This project utilizes several open-source models and libraries. We express our gratitude to the authors and contributors of:

- Diffusers
- LitServe
- Transformers

---

<img src="Readme.svg" width="800" height="400">
