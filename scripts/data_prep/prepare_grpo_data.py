# scripts/data_prep/prepare_grpo.py
import json
import random

def build_grpo_dataset(sft_path, output_path, easy_trap_ratio=0.2):
    grpo_data = []
    
    with open(sft_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        data = json.loads(line)
        instruction = data["instruction"]
        input_text = data["input"]
        output_text = data["output"]
        
        # 1. 原封不动地继承 SFT 的格式 (包含 Hard Negative 和 正样本)
        # 严格对齐 SFT 里的 ChatML 拼接逻辑
        prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        
        # 将原答案 output 作为元数据（metadata）传给 GRPO，供裁判打分时作弊用
        grpo_data.append({
            "prompt": prompt,
            "ground_truth": output_text 
        })
        
        # 2. 策略 B：额外构造 Easy Negative (没有 Context 的情况)
        if random.random() < easy_trap_ratio:
            # 故意把 input_text 设为空字符串
            trap_prompt = f"<|im_start|>user\n{instruction}\n<|im_end|>\n<|im_start|>assistant\n"
            grpo_data.append({
                "prompt": trap_prompt,
                "ground_truth": "抱歉，提供的参考资料中未包含" # 强制其拒答
            })
            
    # 打乱数据顺序
    random.shuffle(grpo_data)
            
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in grpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"✅ GRPO 数据准备完毕！共 {len(grpo_data)} 条。")

if __name__ == "__main__":
    build_grpo_dataset("../../data/processed/rag_sft_dataset_1.jsonl", "../../data/processed/grpo_train.jsonl")