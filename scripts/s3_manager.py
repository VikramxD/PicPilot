import base64
import io
import boto3
from botocore.config import Config
import random
import string
from dotenv import load_dotenv

env = load_dotenv('../config.env')


class ImageService:
    def _init_(self):
        self.s3 = boto3.client(
            "s3",
            config=Config(signature_version="s3v4"),
            aws_access_key_id=env.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=env.AWS_SECRET_ACCESS_KEY,
            region_name=env.AWS_REGION,
        )

    def generate_signed_url(self, file_name: str, exp: int = 1800) -> str:
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": env.AWS_BUCKET_NAME, "Key": file_name},
            ExpiresIn=exp,
        )

    def generate_unique_file_name(self, file) -> str:
        file_name = file.filename
        random_string = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=10)
        )
        file_extension = file_name.split(".")[-1]
        file_real_name = file_name.split(".")[0]
        return f"{file_real_name}-{random_string}.{file_extension}"

    def upload_file(self, file, file_name) -> str:
        self.s3.upload_fileobj(file, env.AWS_BUCKET_NAME, file_name)
        return file_name

    def upload_base64_file(self, base64_file: str, file_name: str) -> str:
        return self.upload_file(io.BytesIO(base64.b64decode(base64_file)), file_name)

    def get_object(self, file_name: str, bucket: str):
        try:
            return self.s3.get_object(Bucket=bucket, Key=file_name)
        except:  # noqa: E722
            return None
