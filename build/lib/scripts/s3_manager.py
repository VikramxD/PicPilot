import base64
import io
import os
import boto3
from botocore.config import Config
import random
import string
from config_settings import settings




class S3ManagerService:
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            config=Config(signature_version="s3v4"),
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
    
    def generate_signed_url(self, file_name: str, exp: int = 43200) -> str:  # 43200 seconds = 12 hours
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.AWS_BUCKET_NAME, "Key": file_name},
            ExpiresIn=exp,
        )

    def generate_unique_file_name(self, file_name: str) -> str:
        random_string = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=10)
        )
        file_extension = "png"
        file_real_name = file_name.split(".")[0]
        return f"{file_real_name}-{random_string}.{file_extension}"

    def upload_file(self, file, file_name) -> str:
        self.s3.upload_fileobj(file, settings.AWS_BUCKET_NAME, file_name)
        return file_name

    def upload_base64_file(self, base64_file: str, file_name: str) -> str:
        return self.upload_file(io.BytesIO(base64.b64decode(base64_file)), file_name)

    def get_object(self, file_name: str, bucket: str):
        try:
            return self.s3.get_object(Bucket=bucket, Key=file_name)
        except self.s3.exceptions.NoSuchKey:
            print(f"The file {file_name} does not exist in the bucket {bucket}.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
