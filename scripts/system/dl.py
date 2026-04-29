import os
import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_ENDPOINT"] = "​hf-mirror.com"
from huggingface_hub import snapshot_download

snapshot_download(repo_id="BAAI/bge-m3", local_dir="D:/develop/models/bge-m3")
snapshot_download(repo_id="BAAI/bge-reranker-base", local_dir="D:/develop/models/bge-reranker-base")