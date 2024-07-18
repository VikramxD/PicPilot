---
license: mit
sdk: docker
emoji: 🚀
colorFrom: blue
colorTo: green
pinned: false
short_description: PicPilot Production Server
---
# 🚀 PicPilot 

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![SDK](https://img.shields.io/badge/sdk-docker-blue.svg)
![Color](https://img.shields.io/badge/color-blue--green-brightgreen.svg)

> PicPilot: Generate Stunning Photography and Craft Visual Narratives in seconds for your Brand

## 📖 Overview

PicPilot is an innovative outpainting pipeline that leverages state-of-the-art Text to Image Models to extend and enhance images. This project has evolved through multiple iterations, addressing challenges and improving output quality at each stage.

### Key Features:
- Advanced segmentation using Segment Anything VIT Huge and YOLOv8l
- High-quality outpainting with Kandinsky-v2.2-decoder-inpaint
- Optimized for NVIDIA A100 40GB GPU
- Customizable prompts and settings
- Batch API support 

## 🏗 Architecture

![image](https://github.com/user-attachments/assets/2961f39b-f554-4c5e-8b62-3cdc30fff46d)

Our pipeline combines multiple AI models to achieve superior outpainting results:
1. **Object Detection**: YOLOv8l provides accurate bounding boxes
2. **Segmentation**: Segment Anything VIT Huge creates precise masks
3. **Outpainting**: Kandinsky-v2.2-decoder-inpaint generates high-quality extended images

## 🛠 Installation

```bash
git clone https://github.com/your-username/picpilot-production-server.git
cd picpilot-production-server
pip install -r requirements.txt
wandb login
huggingface-cli login
cd scripts
```

## 🚀 Usage

Run the main script with your desired parameters:

```bash
python run.py --image_path /path/to/image.jpg \
              --prompt 'Your prompt here' \
              --negative_prompt 'Negative prompt here' \
              --output_dir /path/to/output \
              --mask_dir /path/to/mask \
              --uid unique_id
```

## 🧠 Models

We've experimented with several cutting-edge models:
- [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
- [ControlNet Segmentation](https://huggingface.co/lllyasviel/sd-controlnet-seg)
- [Kandinsky 2.2 Decoder Inpaint](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint)

## 📸 Results

Here are some impressive results from our pipeline:

<div align="center">
  <img src="https://github.com/VikramxD/product_diffusion_api/assets/72499426/1228718b-5ef7-44a1-81f6-2953ffdc767c" width="30%" alt="Cooker Output">
  <img src="https://github.com/VikramxD/product_diffusion_api/assets/72499426/06e12aea-cdc2-4ab8-97e0-be77bc49a238" width="30%" alt="Toaster Output">
  <img src="https://github.com/VikramxD/product_diffusion_api/assets/72499426/65bcd04f-a715-43c3-8928-a9669f8eda85" width="30%" alt="Chair Output">
</div>
<div align="center">
  <img src="https://github.com/VikramxD/product_diffusion_api/assets/72499426/dd6af644-1c07-424a-8ba6-0715a5611094" width="30%" alt="Tent Output">
  <img src="https://github.com/VikramxD/product_diffusion_api/assets/72499426/b1b8c745-deb4-41ff-a93a-77fa06f55cc3" width="30%" alt="Cycle Output">
</div>

## 📊 Experimentation & Improvements

For detailed insights into our experimentation process, check out our [Weights & Biases Report](https://wandb.ai/vikramxd/product_placement_api/reports/Experimentation-Report---Vmlldzo3Mjg1MjQw).

Recent improvements:
- ✅ Deployed model as an API for batch processing
- ✅ Implemented UI using Gradio / Streamlit for visual interaction
- ✅ Integrated image-to-video model pipeline using [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)

## 🎥 Demo Video

Check out our short demo video to see PicPilot in action:

https://github.com/VikramxD/product_diffusion_api/assets/72499426/c935ec2d-cb76-49dd-adae-8aa4feac211e

---

📄 License: MIT

