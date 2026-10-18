"""
Download the published Harvard Dataverse dataset into ./dataset_files.
Requires internet access and: pip install requests
Run: python download_dataset.py
If this fails, use the browser link in README_FIRST.txt.
"""
from pathlib import Path
import requests, sys

PID = "doi:10.7910/DVN/VM4OR3"
BASE = "https://dataverse.harvard.edu"
OUT = Path(__file__).parent / "dataset_files"
OUT.mkdir(exist_ok=True)

def main():
    url = f"{BASE}/api/datasets/:persistentId/versions/:latest"
    try:
        response = requests.get(url, params={"persistentId": PID}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        files = payload["data"]["files"]
        if not files:
            raise RuntimeError("No files listed by the Dataverse API.")
        for entry in files:
            datafile = entry["dataFile"]
            fid = datafile["id"]
            name = datafile.get("filename", f"dataverse_file_{fid}")
            target = OUT / name
            print("Downloading:", name)
            with requests.get(f"{BASE}/api/access/datafile/{fid}", stream=True, timeout=120) as r:
                r.raise_for_status()
                with target.open("wb") as f:
                    for chunk in r.iter_content(1024*1024):
                        if chunk:
                            f.write(chunk)
            print("Saved:", target)
        print("Download complete:", OUT.resolve())
    except Exception as exc:
        print("Automatic download failed:", exc)
        print("Download manually from:")
        print("https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VM4OR3")
        sys.exit(1)

if __name__ == "__main__":
    main()
