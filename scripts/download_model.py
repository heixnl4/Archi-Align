# 在服务器上运行这个简短的 python 脚本
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp/models')
print(f"模型已下载到: {model_dir}")