import json
import os
import sys
# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import clean_and_chunk_docx

# 调用函数处理 Word 文档
file_path = "../../data/raw/外国建筑史.docx"  # 替换为你的 .docx 文件路径
chunks = clean_and_chunk_docx(file_path, chunk_size=500, overlap=50)

# 保存为 JSONL 格式（每行一个 chunk）
output_path = "../../data/processed/test_chunks.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps({"input": chunk}, ensure_ascii=False) + "\n")

print(f"已保存 {len(chunks)} 个文本块到 {output_path}")