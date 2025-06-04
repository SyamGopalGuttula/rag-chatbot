from utils.s3_loader import download_supported_files_from_s3
from utils.file_parser import extract_text_from_file

files = download_supported_files_from_s3()
for path in files:
    print(f"\n📄 Extracting: {path}")
    text = extract_text_from_file(path)
    print(text[:500])  # Preview first 500 chars
