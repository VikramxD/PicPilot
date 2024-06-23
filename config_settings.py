from pydantic import BaseSettings

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str = ''
    AWS_SECRET_ACCESS_KEY: str = ''
    AWS_REGION: str = "ap-south-1"
    AWS_BUCKET_NAME: str="diffusion-model-bucket"

   
   
settings = Settings()

