import os
import requests

# ========== CONFIGURATION ==========
OWNER = "BAMresearch"
REPO = "automatic-sem-image-segmentation"
FOLDER_PATH = "Datasets/Electron Microscopy Images"
BRANCH = "master"
DOWNLOAD_DIR = "Electron_Microscopy_Images"


# ==================================

def download_github_folder(owner, repo, folder_path, branch, download_dir):
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{folder_path}?ref={branch}"
    response = requests.get(api_url)

    if response.status_code != 200:
        print(f"Failed to fetch folder. HTTP {response.status_code}")
        print(response.text)
        return

    items = response.json()
    os.makedirs(download_dir, exist_ok=True)

    for item in items:
        if item["type"] == "file":
            file_url = item["download_url"]
            file_name = os.path.join(download_dir, os.path.basename(item["path"]))
            print(f"⬇️ Downloading {file_name} ...")

            file_data = requests.get(file_url)
            with open(file_name, "wb") as f:
                f.write(file_data.content)
        elif item["type"] == "dir":
            # Recursively download subfolders
            sub_folder = os.path.join(download_dir, os.path.basename(item["path"]))
            download_github_folder(owner, repo, item["path"], branch, sub_folder)

    print(f"✅ Folder '{folder_path}' downloaded successfully to '{download_dir}'")


if __name__ == "__main__":
    download_github_folder(OWNER, REPO, FOLDER_PATH, BRANCH, DOWNLOAD_DIR)
