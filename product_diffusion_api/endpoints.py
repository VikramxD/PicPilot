from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import sdxl_text_to_image
from routers import painting

import logfire
logfire.configure()
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




@app.get('/')
async def root():
    return {
        'message': 'Welcome to the main API Page for the product Photography Application , we provide APIs for building Stunning Product Images with Diffusion Models',
        'description': 'This API provides endpoints for accessing and managing product diffusion data.',
        'version': '1.0.0',
        'author': 'Vikramjeet Singh',
        'contact': {
            'email': 'singh.vikram.1782000@gmail.com',
            'github': 'https://github.com/vikramxD',
            'website': 'https://vikramxd.github.io'
        },
        'license': 'MIT',
    }
    
@app.get("/health")
def check_health():
    return {"status": "ok"}

