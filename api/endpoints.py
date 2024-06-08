from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import (sdxl_text_to_image,painting)
import logfire
import uvicorn 



logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))







app = FastAPI(openapi_url='/api/v1/product-diffusion/openapi.json',docs_url='/api/v1/product_diffusion/docs')
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers=["*"],
    allow_credentials = True
       
)

app.include_router(sdxl_text_to_image.router, prefix='/api/v1/product-diffusion')
app.include_router(painting.router,prefix='/api/v1/product-diffusion')
logfire.instrument_fastapi(app)



@app.get('/')
async def root():
    return {
        'message': 'Welcome to the PICPILOT API Page , Develop Visual Stories for your Brand',
        'description': 'This API provides endpoints for accessing and managing product diffusion data.',
        'version': '1.0.0',
        'author': 'Vikramjeet Singh',
        'contact': {
            'email': 'singh.vikram.1782000@gmail.com',
            'github': 'https://github.com/vikramxD',
            'website': 'https://vikramxd.github.io',
            'peerlist': 'https://peerlist.io/vikramxd'
        },
        'license': 'MIT',
    }
    
@app.get("/health")
def check_health():
    return {"status": "ok"}

<<<<<<< HEAD:api/endpoints.py


uvicorn.run(app, host='127.0.0.1', port=7860)
=======
uvicorn.run(app, host="0.0.0.0", port=8000)

>>>>>>> cceaa9e (chore: Update roi_scale in inpainting.yaml and add gradio UI for SDXL LORA Inpainting):product_diffusion_api/endpoints.py
