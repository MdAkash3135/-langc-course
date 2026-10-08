import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader
)
from dotenv import load_dotenv
load_dotenv()

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"hello this is akash from dhaka bangladesh")
        temp_file_path = temp_file.name
        print(temp_file_path)
load_text_file()