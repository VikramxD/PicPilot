provider "aws" {
  region     = "ap-south-1"
  access_key = var.aws_access_key
  secret_key = var.aws_secret_key
}





resource "aws_s3_bucket" "diffusion_model_bucket" {
  bucket = "diffusion-model-bucket"
  tags = {
    Name    = "Diffusion Model Bucket"
    Task    = "SDXL LORA"
    Product = "Product Diffusion API"
  }

}

resource "aws_s3_bucket_ownership_controls" "s3_bucket_acl_ownership" {
  bucket = aws_s3_bucket.diffusion_model_bucket.id
  rule {
    object_ownership = "ObjectWriter"
  }

}

resource "aws_s3_bucket_public_access_block" "s3_bucket_public_access_block" {
  bucket                  = aws_s3_bucket.diffusion_model_bucket.id
  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = true
  restrict_public_buckets = true
}


resource "aws_s3_bucket_acl" "acl_access" {
  depends_on = [
    aws_s3_bucket_ownership_controls.s3_bucket_acl_ownership,
    aws_s3_bucket_public_access_block.s3_bucket_public_access_block,
  ]

  bucket = aws_s3_bucket.diffusion_model_bucket.id
  acl    = "public-read"
}


