import os
import sys
import json
import urllib.request
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

TARGETS = {
    "1_clapAI_UIT-VSMEC_csv": {
        "url_base": "https://huggingface.co/datasets/clapAI/UIT-VSMEC/resolve/main",
        "files": ["train.csv", "val.csv", "test.csv", "README.md"]
    },
    "2_tridm_UIT-VSMEC_json": {
        "url_base": "https://huggingface.co/datasets/tridm/UIT-VSMEC/resolve/main",
        "files": ["train.json", "valid.json", "test.json", "README.md"]
    },
    "3_SEACrowd_uit_vsmec": {
        "url_base": "https://huggingface.co/datasets/SEACrowd/uit_vsmec/resolve/main",
        "files": ["uit_vsmec.py", "requirements.txt", "LICENSE", "README.md", "__init__.py"],
        "extra_downloads": {
            "raw_data/train.csv": "https://docs.google.com/spreadsheets/export?id=10VYzfK7JLg-vfmqH0UmKX62z_uaXU-Hp&format=csv",
            "raw_data/valid.csv": "https://docs.google.com/spreadsheets/export?id=1EsSFZ94fj2yTvFKO6EyxM0wBRcG0s1KE&format=csv",
            "raw_data/test.csv": "https://docs.google.com/spreadsheets/export?id=1D16FCKKgJ0T6t2aSA3biWVwvD9fa4G9a&format=csv"
        }
    }
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading: {url} -> {dest_path}")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())
    size = os.path.getsize(dest_path)
    print(f"  [OK] Saved ({size:,} bytes)")

def main():
    print(f"Saving datasets to: {DATA_DIR}")
    for folder_name, info in TARGETS.items():
        folder_path = os.path.join(DATA_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"\n==========================================")
        print(f"Processing: {folder_name}")
        print(f"==========================================")
        
        url_base = info.get("url_base")
        for filename in info.get("files", []):
            url = f"{url_base}/{filename}"
            dest = os.path.join(folder_path, filename)
            download_file(url, dest)
            
        if "extra_downloads" in info:
            print("\nDownloading SEACrowd raw dataset source files (from original Google Sheets)...")
            for subpath, url in info["extra_downloads"].items():
                dest = os.path.join(folder_path, subpath)
                download_file(url, dest)

    print("\n==========================================")
    print("ALL DOWNLOADS COMPLETE! VERIFYING DATA...")
    print("==========================================")

    # Verify clapAI
    c_train = pd.read_csv(os.path.join(DATA_DIR, "1_clapAI_UIT-VSMEC_csv", "train.csv"))
    c_val = pd.read_csv(os.path.join(DATA_DIR, "1_clapAI_UIT-VSMEC_csv", "val.csv"))
    c_test = pd.read_csv(os.path.join(DATA_DIR, "1_clapAI_UIT-VSMEC_csv", "test.csv"))
    print(f"1. clapAI (CSV):")
    print(f"   - train: {len(c_train)} rows | cols: {list(c_train.columns)}")
    print(f"   - val:   {len(c_val)} rows")
    print(f"   - test:  {len(c_test)} rows")
    print(f"   - labels distribution in train:\n{c_train['label'].value_counts().to_dict()}")

    # Verify tridm
    with open(os.path.join(DATA_DIR, "2_tridm_UIT-VSMEC_json", "train.json"), "r", encoding="utf-8") as f:
        t_train = json.load(f)
    with open(os.path.join(DATA_DIR, "2_tridm_UIT-VSMEC_json", "valid.json"), "r", encoding="utf-8") as f:
        t_valid = json.load(f)
    with open(os.path.join(DATA_DIR, "2_tridm_UIT-VSMEC_json", "test.json"), "r", encoding="utf-8") as f:
        t_test = json.load(f)
    print(f"\n2. tridm (JSON):")
    print(f"   - train: {len(t_train)} items | sample keys: {list(t_train[0].keys())}")
    print(f"   - valid: {len(t_valid)} items")
    print(f"   - test:  {len(t_test)} items")

    # Verify SEACrowd
    s_train = pd.read_csv(os.path.join(DATA_DIR, "3_SEACrowd_uit_vsmec", "raw_data", "train.csv"))
    s_valid = pd.read_csv(os.path.join(DATA_DIR, "3_SEACrowd_uit_vsmec", "raw_data", "valid.csv"))
    s_test = pd.read_csv(os.path.join(DATA_DIR, "3_SEACrowd_uit_vsmec", "raw_data", "test.csv"))
    print(f"\n3. SEACrowd (Python Script + Raw CSVs):")
    print(f"   - train: {len(s_train)} rows | cols: {list(s_train.columns)}")
    print(f"   - valid: {len(s_valid)} rows")
    print(f"   - test:  {len(s_test)} rows")

if __name__ == "__main__":
    main()
