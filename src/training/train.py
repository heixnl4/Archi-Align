import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

def process_dataset(example, tokenizer, max_length=1024):
    """
    底层逻辑：将 JSONL 转化为带有 Loss Mask 的 Token IDs
    """
    # Qwen 的原生 ChatML 格式模板
    instruction = f"<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
    response = f"{example['output']}<|im_end|>\n"
    
    # 1. 分别对 Instruction 和 Response 进行 Tokenize
    instr_ids = tokenizer(instruction, add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    
    # 2. 拼接完整的 input_ids
    input_ids = instr_ids + resp_ids
    
    # 3. 构造 labels：Instruction 部分全部设为 -100（不计算 Loss），Response 部分保留原 ID
    labels = [-100] * len(instr_ids) + resp_ids
    
    # 4. 截断处理（防止超出显存）
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids) # 1 表示需要模型关注
    }

def load_model_and_tokenizer(model_path):
    """
    加载模型，应用显存优化机制，并注入 LoRA 旁路矩阵
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 很多大模型没有 pad_token，通常借用 eos_token 以防止 batch 拼接时出错
    tokenizer.pad_token = tokenizer.eos_token 

    # 显存优化：使用 bfloat16 精度加载基座模型（需要 3090/4090 或 A100 支持）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto", # 自动分配显存
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    
    # 开启梯度检查点（牺牲 20% 左右的计算速度，换取近 50% 的显存节省），Infra 必备技巧
    model.gradient_checkpointing_enable()
    
    # LoRA 配置：只更新注意力机制的 q_proj 和 v_proj
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                 # 秩，决定旁路矩阵的参数量
        lora_alpha=16,       # 缩放系数
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"] 
    )
    
    # 注入 LoRA 矩阵，此时模型的可训练参数将从 70亿 骤降至 千万级别
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    # 这里的路径是在云服务器上的绝对路径
    model_path = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct" 
    data_path = "../../data/processed/rag_sft_dataset.jsonl"
    
    print(">>> 1. 初始化模型与分词器...")
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    print(">>> 2. 加载与处理数据集...")
    raw_dataset = load_dataset("json", data_files=data_path, split="train")
    
    # 使用 map 函数并行处理数据，转为底层的 Token IDs
    train_dataset = raw_dataset.map(
        lambda x: process_dataset(x, tokenizer),
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing & Masking"
    )
    
    print(">>> 3. 配置训练超参数...")
    training_args = TrainingArguments(
        output_dir="../../outputs/Qwen-Arch-LoRA",
        per_device_train_batch_size=2, # 取决于显存，4090 一般可以开到 2 或 4
        gradient_accumulation_steps=4, # 累积 4 步才更新一次梯度，等效 Batch Size = 8
        learning_rate=2e-4,            # LoRA 经典学习率
        num_train_epochs=3,            # 过几遍数据集
        logging_steps=10,
        save_strategy="epoch",         # 每个 epoch 保存一次权重
        bf16=True,                     # 开启混合精度加速
        optim="adamw_torch"
    )
    
    # DataCollator 负责在组成 batch 时，动态地用 pad_token 补齐长度不一的句子
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model, 
        label_pad_token_id=-100 # 确保补齐的部分也不参与 loss 计算
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    print(">>> 4. 开始训练！")
    trainer.train()
    
    print(">>> 5. 保存最终 LoRA 权重...")
    trainer.model.save_pretrained("../../outputs/Qwen-Arch-LoRA/final")
    tokenizer.save_pretrained("../../outputs/Qwen-Arch-LoRA/final")

if __name__ == "__main__":
    main()