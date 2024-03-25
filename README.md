# Kandinsky Inpainting Project

This project uses the Kandinsky inpainting pipeline and a mask generation script to perform inpainting on an image. The mask is generated using the YOLOv8s seg model from the Ultralytics library for mask generation. The mask is then inverted and used in the Kandinsky inpainting pipeline

## Installation

To install the necessary requirements, you can use pip:

```bash
pip install -r requirements.txt
wandb login
huggingface-cli login
cd scripts
```

This will install all necessary libraries for this project, including PIL, numpy, Ultralytics wandb, diffuser etc.

### cd in to scripts
specify/create the folders manually before this
```bash
python run.py --image_path /path/to/image.jpg --prompt 'prompt' --negative_prompt 'negative prompt' --output_dir /path/to/output --mask_dir /path/to/mask --uid unique_id


```
### Some Experiments
Here are some of my experiments with the following models
 - https://huggingface.co/runwayml/stable-diffusion-inpainting
 - https://huggingface.co/lllyasviel/sd-controlnet-seg
 - kandinsky-community/kandinsky-2-2-decoder-inpaint
 - https://wandb.ai/vikramxd/product_placement_api/reports/Generated-Image-Pipeline-Call-1-24-03-22-21-45-35---Vmlldzo3MjYxMzcy

![cooker_output](https://github.com/VikramxD/product_diffusion_api/assets/72499426/1228718b-5ef7-44a1-81f6-2953ffdc767c)
![toaster_output](https://github.com/VikramxD/product_diffusion_api/assets/72499426/06e12aea-cdc2-4ab8-97e0-be77bc49a238)
![tent_output](https://github.com/VikramxD/product_diffusion_api/assets/72499426/bb4a6af4-7652-4722-8bf6-88f6fbceefff)
