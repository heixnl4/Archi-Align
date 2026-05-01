import json
import random

def prepare_grpo_dataset(input_path, output_path, trap_ratio=0.3):
    """
    trap_ratio: 陷阱题（无Context）的比例
    """
    grpo_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        item = json.loads(line)
        question = item.get("instruction", "")
        context = item.get("input", "")
        
        # 1. 正常样本：提供正确的 Context
        grpo_data.append({
            "prompt": f"请根据参考资料回答问题。要求：回答必须严谨，并使用 [1] 标注出处。\n参考资料：{context}\n问题：{question}",
            "is_trap": False
        })
        
        # 2. 构造陷阱样本：随机抽取问题，但把 Context 设为空
        if random.random() < trap_ratio:
            grpo_data.append({
                "prompt": f"请根据参考资料回答问题。要求：回答必须严谨，并使用 [1] 标注出处。\n参考资料：无相关信息。\n问题：{question}",
                "is_trap": True
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in grpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ GRPO 数据集已生成：{len(grpo_data)} 条，包含约 {int(len(grpo_data)*trap_ratio)} 条陷阱题。")

if __name__ == "__main__":
    prepare_grpo_dataset("../../data/processed/rag_sft_dataset_1.jsonl", "../../data/processed/grpo_train_data.jsonl")