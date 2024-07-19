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

PicPilot is a scalable solution that leverages state-of-the-art Text to Image Models to extend and enhance images. This project has evolved through multiple iterations, addressing challenges and improving output quality at each stage.

### Key Features:
- segmentation using Segment Anything VIT Huge and YOLOv8s
- High-quality outpainting with Controlnet + ZoeDepth
- stable video diffusion support 
- Batch API support and EventDriven Queue Support
- Logging and Telemetry using LogFire
   

## 🏗 Architecture

![image](https://github.com/user-attachments/assets/2961f39b-f554-4c5e-8b62-3cdc30fff46d)

Current Pipeline 
1. **Object Detection**: YOLOv8l provides accurate bounding boxes
2. **Segmentation**: Segment Anything VIT Huge creates precise masks with ROI extension
3. **Outpainting**: Controlnet Zoe Depth + Realistic Vision XL
4. **I2V GenXL**: Image to Video Generation


## 🧠 Models used 


- [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
- [Kandinsky 2.2 Decoder Inpaint](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint)
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Controlnet-Inpaint Dreamer](https://huggingface.co/destitech/controlnet-inpaint-dreamer-sdxl)
- [Controlnet Zoe Depth](https://huggingface.co/diffusers/controlnet-zoe-depth-sdxl-1.0)
- [Realistic Vision XL](https://huggingface.co/OzzyGT/RealVisXL_V4.0_inpainting)
- [I2V GenXL](https://huggingface.co/ali-vilab/i2vgen-xl)

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
- ✅ Implemented UI using Gradio 
- ✅ Integrated image-to-video model pipeline using [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and 

## 🎥 Sample Video

Check out our short demo video to see PicPilot in action:

https://github.com/VikramxD/product_diffusion_api/assets/72499426/c935ec2d-cb76-49dd-adae-8aa4feac211e

---

📄 License: MIT

