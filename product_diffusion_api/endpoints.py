from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI(openapi_url='/api/v1/product-diffusion/openapi.json',docs_url='/api/v1/product_diffusion/docs')
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers=["*"],
    allow_credentials = True
    
)

app.include_router(sdxl_text_to_image.router, prefix='/api/v1/product-diffusion')
app.include_router()


@app.get('/')
async def root():
    return {'message: Product Diffusion API'}