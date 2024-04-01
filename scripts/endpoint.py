from fastapi import FastAPI, UploadFile
from PIL import Image
from io import BytesIO
from fastapi import File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from mask_generator import extend_image, invert_mask
from segment_everything import generate_mask_from_bbox
from models import kandinsky_inpainting_inference


app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/outpaint")
def outpainting(prompt,negative_prompt, image:UploadFile= File(...),target_width: int = 2560, target_height: int = 1440,roi=0.6):
    
    image = Image.open(BytesIO(image.read()))
    image = extend_image(image, target_width, target_height, roi)
    
    mask = generate_mask_from_bbox(image)
    mask = invert_mask(mask)
    mask = Image.fromarray(mask)
    image = kandinsky_inpainting_inference(prompt=prompt, negative_prompt=negative_prompt,image=image, mask=mask)
    return FileResponse(image, media_type="image/png")