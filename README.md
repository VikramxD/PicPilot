# Outpainting Pipeline

- Intial Version of the project used a combination of Yolov8s Segmentation model to provide a rough mask , which was then inverted for Outpainting to use with Models like stable diffusion inpainting model from Runway along with ControlNet to control the outpainted generated output , 
- There were blockers in that approach as first the mask , was of a poor quality with rough edges which were messing with the outpainting output , even with techniques like blurring the mask the output quality was poor initailly with stable diffusion models
- To address this , changes like detecting the ROI of the object in focus in addition to extending and resizing the image was done , the model for segmentation was upgraded to Segment Anything VIT Huge with yolov8l model , providing the bounding boxes for the Box prompt which was then inverted for outpainting 
- The model was changed kandinsky-v2.2-decoder-inpaint with 800 inference steps , a guidence scale of 5.0 to 7.5 and then the following results were achieved
- GPU used Nvidia A100 40GB 


## ARCHITECTURE

![Architecture drawio](https://github.com/VikramxD/product_diffusion_api/assets/72499426/5a2e8b47-5a77-485b-b20c-0bca0928cb8a)


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
### MODELS USED 
EXPERIMENTATION WITH THE FOLLOWING models
 - https://huggingface.co/runwayml/stable-diffusion-inpainting
 - https://huggingface.co/lllyasviel/sd-controlnet-seg
 - https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint

### WEIGHTS AND BIASES EXPERIMENTATION REPORT 

[WANDB REPORT](https://wandb.ai/vikramxd/product_placement_api/reports/Experimentation-Report---Vmlldzo3Mjg1MjQw)

![cooker_output](https://github.com/VikramxD/product_diffusion_api/assets/72499426/1228718b-5ef7-44a1-81f6-2953ffdc767c)
![toaster_output](https://github.com/VikramxD/product_diffusion_api/assets/72499426/06e12aea-cdc2-4ab8-97e0-be77bc49a238)
![chair](https://github.com/VikramxD/product_diffusion_api/assets/72499426/65bcd04f-a715-43c3-8928-a9669f8eda85)
![Generated Image Pipeline Call 1](https://github.com/VikramxD/product_diffusion_api/assets/72499426/dd6af644-1c07-424a-8ba6-0715a5611094)
![Generated Image Pipeline Call (1)](https://github.com/VikramxD/product_diffusion_api/assets/72499426/b1b8c745-deb4-41ff-a93a-77fa06f55cc3)

## Some Improvements 
- Working on API to deploy this model in batch mode adding loggers from prompt and generated output 
- Implementation of UI in Gradio / Streamlit for checking the model out in visual way
- Experimenting with image to video model pipeline to generate a video output thinking of using (https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) model for this
