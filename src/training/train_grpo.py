# train_grpo.py
import re
import torch
from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel


# ================= 1. 定义奖励函数 (Reward Functions) =================

def format_reward_func(completions, ground_truth, **kwargs):
    """
    格式裁判：只对正样本检查 [x] 引用标记。
    如果是陷阱题（拒答题），不强求加引用。
    """
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        is_trap = "抱歉" in gt
        if not is_trap:
            # 正样本：必须有引用才给分
            if re.search(r'\[\d+\]', comp):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            # 陷阱题：不考核引用格式，给个默认基础分
            rewards.append(0.5)
    return rewards

def faithfulness_reward_func(completions, ground_truth, **kwargs):
    """
    忠实度裁判：开了上帝视角，比对 SFT 的标签。
    """
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        is_trap = "抱歉" in gt # 判断这道题是不是 SFT 里的陷阱题
        
        if is_trap:
            # 陷阱题，模型必须说“抱歉”
            if "抱歉" in comp or "未包含" in comp or "未提及" in comp:
                rewards.append(2.0) # 听话拒答，重赏！
            else:
                rewards.append(-2.0) # 强行瞎编，严惩！
        else:
            # 正常题，模型绝不能随意说“抱歉”
            if "抱歉" in comp or "未包含" in comp:
                rewards.append(-2.0) # 资料里明明有，它却不回答，严惩！
            else:
                rewards.append(0.5) # 正常作答，给基础分
    return rewards

# ================= 2. 配置并启动 GRPO 训练 =================
def main():
    # 1. 明确区分两个路径
    base_model_path = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"  # 几十GB的原模型
    sft_lora_path = "./outputs/Qwen-Arch-LoRA/final"                # 几百MB的SFT增量权重
    data_path = "../../data/processed/grpo_train.jsonl"

    # 2. 加载原模型的 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(">>> 1. 正在加载 7B 原始基座模型...")
    # 3. 加载原始基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16, # 4090 支持 bfloat16，防溢出
        attn_implementation="sdpa",
        device_map="auto"
    )

    print(">>> 2. 正在挂载 SFT LoRA 并进行物理合并...")

    # 4. 把 SFT 的增量权重挂载上去
    model = PeftModel.from_pretrained(base_model, sft_lora_path)
    # 【核心操作】：将 LoRA 权重彻底融入基座模型的线性层中，并卸载独立的外挂结构
    model = model.merge_and_unload()
    # 开启梯度检查点省显存
    model.gradient_checkpointing_enable()
    print(">>> ✅ SFT 记忆已永久固化入基座模型！")

    print(">>> 🔧 为模型加装生成长度锁 (防止 GRPO 探索时爆显存)...")
    model.generation_config.max_new_tokens = 128

    # 5. 准备数据集
    print(">>> 3. 加载 GRPO 数据集...")
    dataset = load_dataset("json", data_files=data_path, split="train")

    # 6. GRPO 专属配置 (保持不变)
    training_args = GRPOConfig(
        output_dir="../../outputs/Qwen-Arch-GRPO",
        learning_rate=5e-6,           # 强化学习的学习率必须极小！通常是 SFT 的十分之一
        per_device_train_batch_size=1,   # 4090 显存紧张，Batch Size 设为 1 
        gradient_accumulation_steps=4,
        num_generations=4,            # GRPO 核心：对同一个问题生成 4 个不同答案进行组内对比
        save_steps=50,
        logging_steps=10,
        bf16=True,
        report_to="none"              # 关闭 wandb 避免本地报错
    )

    # 7. 为 GRPO 阶段初始化一个【全新】的 LoRA 配置
    # 这个新的 LoRA 就是模型用来试错、学习强化学习规则的“草稿纸”
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # 建议多加几个 target
        task_type="CAUSAL_LM",
    )

    print(">>> 3. 初始化 GRPO Trainer...")
    # 8. 把刚刚合并好的 model 丢给 Trainer，配上新的 peft_config
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, faithfulness_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print(">>> 开始 GRPO 强化学习对齐训练...")
    trainer.train()

    print(">>> 保存最终 GRPO LoRA 权重...")
    trainer.model.save_pretrained("../../outputs/Qwen-Arch-GRPO/final")
    tokenizer.save_pretrained("../../outputs/Qwen-Arch-GRPO/final")

if __name__ == "__main__":
    main()