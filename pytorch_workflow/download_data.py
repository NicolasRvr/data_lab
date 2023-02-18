import os 
import zipfile
from pathlib import Path

import requests

def download_data(source: str,
                  destination: str,
                  remove_source: bool=True):
    data_path = Path("data")
    image_path = data_path / destination
    
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping downloading")
    else: 
        print(f"[INFO] Did not found {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
    
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} data...")
            f.write(request.content)
            
        with zipfile.ZipFile(data_path / target_file, 'r') as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...")
            zip_ref.extractall(image_path)
            
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path