# train_grpo.py
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel


# ================= 1. 定义奖励函数 (Reward Functions) =================

def format_reward_func(completions, **kwargs):
    """
    格式奖励：检查模型回复中是否包含了引用标记 [x]
    """
    rewards = []
    # completions 是模型生成的多个候选回答列表
    for comp in completions:
        # 使用正则查找形如 [1], [2] 的标记
        if re.search(r'\[\d+\]', comp):
            rewards.append(1.0) # 格式正确，给奖励
        else:
            rewards.append(0.0) # 没加引用，不给分
    return rewards

def faithfulness_reward_func(prompts, completions, **kwargs):
    """
    忠实度与防幻觉裁判：严格对齐 SFT 样本 1
    核心目标：当 Context 为空或无用时，模型必须输出你 SFT 里规定的标准拒答话术。
    """
    rewards = []
    
    # 这里的触发条件取决于你准备 GRPO 数据时，怎么表示“无参考资料”
    # 假设你的 prompt 里包含了 "参考资料：无相关信息。" 
    trap_trigger = "参考资料：无相关信息" 
    
    # 你的 SFT 标准拒答话术（取核心关键词组合进行匹配，增加容错率）
    standard_refusal_1 = "抱歉，提供的参考资料中未包含"
    
    for prompt, comp in zip(prompts, completions):
        # 1. 遇到陷阱题（无 Context）
        if trap_trigger in prompt:
            # 严格检查是否使用了 SFT 教的标准拒答话术
            if standard_refusal_1 in comp:
                rewards.append(2.0) # 完美对齐 SFT 价值观，给最高分！
            else:
                rewards.append(-2.0) # 没按 SFT 教的规矩拒答（或者胡编乱造了），严惩！
                
        # 2. 遇到正常题（有 Context）
        else:
            # 基础分。这里可以进一步扩展，比如检查 comp 长度是否合理
            # 但在基础的防幻觉对齐中，只要它在正常题里不胡乱说“抱歉”，就给基础分
            if standard_refusal_1 not in comp:
                rewards.append(0.5)
            else:
                rewards.append(-1.0) # 资料里明明有，它却说没有，扣分！
                
    return rewards

# ================= 2. 准备训练数据 =================
def load_grpo_dataset():
    """
    GRPO 需要的格式非常简单，只需要一列 'prompt' 即可。
    这里构造几条典型的 RAG 数据供演示。
    """
    data = [
        # 正向样本：有 Context
        {"prompt": "问题：草莓山住宅是什么风格？\n参考资料：1750年，沃波尔将其草莓山住宅饰以哥特式风格。"},
        # 负向样本（陷阱题）：无 Context，测试拒答能力
        {"prompt": "问题：现代鸟巢体育馆的设计师是谁？\n参考资料：无"},
    ]
    # 在实际项目中，你要把你的 JSONL 数据加载进来，并拼装成这样的 Prompt
    return Dataset.from_list(data)

# ================= 3. 配置并启动 GRPO 训练 =================
def main():
    # 1. 明确区分两个路径
    base_model_path = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"  # 几十GB的原模型
    sft_lora_path = "./outputs/Qwen-Arch-LoRA/final"                # 几百MB的SFT增量权重

    # 2. 加载原模型的 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(">>> 1. 正在加载 7B 原始基座模型...")
    # 3. 加载原始基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto"
    )

    print(">>> 2. 正在挂载 SFT LoRA 并进行物理合并...")

    # 4. 把 SFT 的增量权重挂载上去
    model = PeftModel.from_pretrained(base_model, sft_lora_path)
    # 【核心操作】：将 LoRA 权重彻底融入基座模型的线性层中，并卸载独立的外挂结构
    model = model.merge_and_unload()
    print(">>> ✅ SFT 记忆已永久固化入基座模型！")

    # 5. 准备数据集
    dataset = load_grpo_dataset()

    # 6. GRPO 专属配置 (保持不变)
    training_args = GRPOConfig(
        output_dir="../../outputs/Qwen-Arch-GRPO",
        learning_rate=5e-6,           # 强化学习的学习率必须极小！通常是 SFT 的十分之一
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_prompt_length=256,
        max_completion_length=128,
        num_generations=4,            # GRPO 核心：对同一个问题生成 4 个不同答案进行组内对比
        save_steps=50,
        logging_steps=10,
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

if __name__ == "__main__":
    main()