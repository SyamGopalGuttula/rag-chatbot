from utils.s3_loader import download_supported_files_from_s3

paths = download_supported_files_from_s3()
print("Downloaded files:")
for p in paths:
    print(" -", p)
