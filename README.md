# Kandinsky Inpainting Project

This project uses the Kandinsky inpainting pipeline and a mask generation script to perform inpainting on an image. The mask is generated using the YOLO model from the Ultralytics library for object detection. The mask is then inverted and used in the Kandinsky inpainting pipeline.

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
Here are some of my experiments with controlnet inpaint runway stable diffusion models and finally Kandinsky 2.2 Inpaint Model
https://wandb.ai/vikramxd/product_placement_api/reports/Generated-Image-Pipeline-Call-1-24-03-22-21-45-35---Vmlldzo3MjYxMzcy

![Image](https://github.com/VikramxD/product_diffusion_api/assets/72499426/44a91907-40c3-4f8c-9e42-979c09f58da2)
![Generated Image](https://github.com/VikramxD/product_diffusion_api/assets/72499426/e02ae767-97b3-404d-a27a-56bcbb249d93)
![Generated Image Pipeline Call ](https://github.com/VikramxD/product_diffusion_api/assets/72499426/895823d9-0060-4666-962c-45c2b36a8993)
