import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def download_file_from_s3():
    s3 = boto3.client('s3')
    bucket = os.getenv("S3_BUCKET_NAME")
    key = os.getenv("S3_KEY")
    local_path = os.getenv("LOCAL_PATH")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    print(f"Downloading {key} from S3 bucket {bucket}...")
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded to {local_path}")
