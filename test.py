from utils.s3_loader import download_supported_files_from_s3
from utils.file_parser import extract_text_from_file
from utils.chunker import chunk_text
from vectorstore.chroma_db import embed_and_store
import os

files = download_supported_files_from_s3()

for path in files:
    print(f"\n📄 Processing: {path}")
    text = extract_text_from_file(path)
    chunks = chunk_text(text)

    source_id = os.path.splitext(os.path.basename(path))[0]  # filename without ext
    embed_and_store(chunks, source_id)
