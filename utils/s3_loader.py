import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def download_supported_files_from_s3():
    s3 = boto3.client('s3')
    bucket = os.getenv("S3_BUCKET_NAME")
    allowed_extensions = ('.pdf', '.txt', '.docx')

    paginator = s3.get_paginator('list_objects_v2')
    result = paginator.paginate(Bucket=bucket)

    local_paths = []

    for page in result:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith(allowed_extensions):
                filename = os.path.basename(key)
                local_path = f"tmp/{filename}"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                print(f"📥 Downloading {key} to {local_path}...")
                s3.download_file(bucket, key, local_path)
                local_paths.append(local_path)

    return local_paths
