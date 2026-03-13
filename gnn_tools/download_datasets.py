"""
This script is responsible for downloading and extracting datasets for graph neural network experiments.
It defines a dictionary of datasets with their URLs, filenames, and extraction requirements.
"""

import os
import tarfile
import zipfile

# download_datasets.py

import urllib.request

# Example : 
# DATASETS = {
#     "Cora": {
#         "url": "https://linqs-data.soe.ucsc.edu/public/linqs/cora.tgz",
#         "filename": "cora.tgz",
#         "extract": True
#     },
#     "Citeseer": {
#         "url": "https://linqs-data.soe.ucsc.edu/public/linqs/citeseer.tgz",
#         "filename": "citeseer.tgz",
#         "extract": True
#     },
#     "Pubmed": {
#         "url": "https://linqs-data.soe.ucsc.edu/public/linqs/pubmed.tgz",
#         "filename": "pubmed.tgz",
#         "extract": True
#     },
#     "Reddit": {
#         "url": "https://snap.stanford.edu/graphsage/reddit.zip",
#         "filename": "reddit.zip",
#         "extract": True
#     },
#     "KarateClub": {
#         "url": "https://networkrepository.com/karate.zip",
#         "filename": "karate.zip",
#         "extract": True
#     }
# }

DATASETS = {}

DATA_DIR = "data" #name of the directory where datasets will be stored

def download_and_extract(dataset_name, info):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, info["filename"])
    if not os.path.exists(file_path):
        print(f"Downloading {dataset_name}...")
        urllib.request.urlretrieve(info["url"], file_path)
    else:
        print(f"{dataset_name} already downloaded.")
    if info["extract"]:
        print(f"Extracting {dataset_name}...")
        if file_path.endswith(".tgz") or file_path.endswith(".tar.gz"):
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(DATA_DIR)
        elif file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(DATA_DIR)
    print(f"{dataset_name} ready.")

if __name__ == "__main__":
    for name, info in DATASETS.items():
        download_and_extract(name, info)