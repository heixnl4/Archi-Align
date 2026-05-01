import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def generate_response(model, tokenizer, instruction, input_text):
    """构建与训练时完全一致的 Prompt 模板进行推理"""
    prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad(): # 推理时不需要计算梯度，省显存
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1, # 设定极低的温度，让模型输出更稳定严谨的知识
            repetition_penalty=1.1
        )
    # 只截取 assistant 回复的部分
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

def main():
    base_model_path = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
    lora_path = "../../outputs/Qwen-Arch-LoRA/final"

    print(">>> 1. 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(">>> 2. 加载原始基座模型 (Base Model)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="sdpa"
    )

    # 找一条 RAG 场景下的测试数据（可以是训练集里的，也可以是全新的建筑史文本）
    test_instruction = "现代主义建筑的三条历史线索是什么？"
    test_input = "（这里填入你 Word 文档里对应的一段原文文本...）"

    print(test_instruction)

    print("\n【Base 模型原始输出】:")
    base_out = generate_response(base_model, tokenizer, test_instruction, test_input)
    print(base_out)
    print("-" * 50)

    print(">>> 3. 动态挂载 LoRA 补丁 (Fine-tuned Model)...")
    # PEFT 库的魔法：直接在内存中把 LoRA 权重和基座模型融为一体
    finetuned_model = PeftModel.from_pretrained(base_model, lora_path)

    print("\n【微调后模型输出】:")
    ft_out = generate_response(finetuned_model, tokenizer, test_instruction, test_input)
    print(ft_out)

if __name__ == "__main__":
    main()