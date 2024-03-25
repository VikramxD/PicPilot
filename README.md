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

This will install all necessary libraries for this project, including PIL , Diffusers , Segment Anything, wandb ,

```bash
python run.py --image_path /path/to/image.jpg --prompt 'prompt' --negative_prompt 'negative prompt' --output_dir /path/to/output --mask_dir /path/to/mask --uid unique_id
```










```
### Some Experiments
Here are some of my experiments with the following models
 - https://huggingface.co/runwayml/stable-diffusion-inpainting
 - https://huggingface.co/lllyasviel/sd-controlnet-seg
 - https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint




![cooker_output](https://github.com/VikramxD/product_diffusion_api/assets/72499426/1228718b-5ef7-44a1-81f6-2953ffdc767c)
![toaster_output](https://github.com/VikramxD/product_diffusion_api/assets/72499426/06e12aea-cdc2-4ab8-97e0-be77bc49a238)
!![Generated Image Pipeline Call 1](https://github.com/VikramxD/product_diffusion_api/assets/72499426/2e7a804f-482c-4807-897e-1aa02b2fd37f)
![chair](https://github.com/VikramxD/product_diffusion_api/assets/72499426/65bcd04f-a715-43c3-8928-a9669f8eda85)
![Generated Image Pipeline Call 1](https://github.com/VikramxD/product_diffusion_api/assets/72499426/dd6af644-1c07-424a-8ba6-0715a5611094)
![Generated Image Pipeline Call (1)](https://github.com/VikramxD/product_diffusion_api/assets/72499426/b1b8c745-deb4-41ff-a93a-77fa06f55cc3)


