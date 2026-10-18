#!/usr/bin/env python3
"""
Download files from the Harvard Dataverse dataset DOI into ./dataset_files.
Requires: pip install requests
Run: python download_dataset.py
"""
from pathlib import Path
import requests, sys

PID = "doi:10.7910/DVN/VM4OR3"
BASE = "https://dataverse.harvard.edu"
OUT = Path(__file__).parent / "dataset_files"
OUT.mkdir(exist_ok=True)

def main():
    api = f"{BASE}/api/datasets/:persistentId/versions/:latest"
    try:
        r = requests.get(api, params={"persistentId": PID}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        files = payload["data"]["files"]
        if not files:
            raise RuntimeError("The API returned no files.")
        print(f"Found {len(files)} file(s).")
        for item in files:
            datafile = item["dataFile"]
            file_id = datafile["id"]
            name = datafile.get("filename") or f"dataverse_file_{file_id}"
            target = OUT / name
            print(f"Downloading {name} ...")
            resp = requests.get(f"{BASE}/api/access/datafile/{file_id}", timeout=120, stream=True)
            resp.raise_for_status()
            with target.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"Saved: {target}")
        print(f"\nDone. Files are in: {OUT.resolve()}")
    except Exception as e:
        print("\nAutomatic download failed:", e)
        print("Open the dataset page and download the published file manually:")
        print("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VM4OR3")
        sys.exit(1)

if __name__ == "__main__":
    main()
