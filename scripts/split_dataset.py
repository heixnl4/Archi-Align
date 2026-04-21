# File: scripts/split_dataset.py

import json
import random

def split_data(input_file, train_file, val_file, val_size=50):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
        
    # 打乱数据集（设定固定的随机种子，保证每次切分结果一致）
    random.seed(42)
    random.shuffle(data)
    
    val_data = data[:val_size]
    train_data = data[val_size:]
    
    # 写入训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    # 写入验证集
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"✅ 数据集切分完成！")
    print(f"原始数据总量: {len(data)}")
    print(f"训练集 ({train_file}): {len(train_data)} 条")
    print(f"验证集 ({val_file}): {len(val_data)} 条")

if __name__ == "__main__":
    # 请确保路径与你服务器上的实际结构匹配
    input_path = "../data/processed/rag_sft_dataset.jsonl"
    train_path = "../data/processed/train_sft.jsonl"
    val_path = "../data/processed/val_sft.jsonl"
    
    split_data(input_path, train_path, val_path)