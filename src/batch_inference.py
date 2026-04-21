# File: src/batch_inference.py

import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    base_model_path = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
    lora_path = "./outputs/Qwen-Arch-LoRA/final"
    val_data_path = "./data/processed/val_sft.jsonl"
    output_path = "./data/processed/eval_predictions.jsonl"

    print(">>> 1. 初始化 Tokenizer 与模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map={"": 0}, # 强制挂载在主卡，防止 meta tensor 报错
        trust_remote_code=True,
        attn_implementation="sdpa" # 使用原生加速，绕过 Flash-Attention
    )
    
    print(">>> 2. 挂载 LoRA 权重...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval() # 切换到评估模式

    print(">>> 3. 加载验证集数据...")
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_data = [json.loads(line) for line in f]

    results = []
    print(">>> 4. 开始批量推理...")
    # 使用 tqdm 显示进度条
    for item in tqdm(val_data):
        instruction = item["instruction"]
        input_text = item["input"]
        ground_truth = item["output"]
        
        # 严格遵守 ChatML 模板格式
        prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1, # 极低温度，保证输出的事实严谨性
                repetition_penalty=1.1
            )
            
        # 截取新生成的部分
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        # 组装给大模型裁判的数据结构
        results.append({
            "instruction": instruction,
            "input": input_text,
            "ground_truth": ground_truth,
            "prediction": response.strip()
        })

    # 将预测结果落盘
    with open(output_path, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            
    print(f"\n🎉 批量推理完成！结果已保存至 {output_path}")

if __name__ == "__main__":
    main()