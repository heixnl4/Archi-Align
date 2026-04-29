import json
import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
aclient = AsyncOpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL"))

JUDGE_PROMPT = """你是一个严苛的 AI 算法评测专家。
你需要根据提供的【用户问题】、【参考文本】和【标准答案】，对微调大模型的【预测答案】进行打分。

请从以下三个维度进行 1-5 分的评分（5分为完美，1分为极差）：
1. faithfulness (事实一致性): 预测答案是否完全基于参考文本？是否出现了幻觉（捏造文本中不存在的事实）？如果文本不足以回答，模型是否正确地执行了拒答？
2. instruction_following (指令遵循度): 模型是否直接回答了问题，而没有使用啰嗦的开头（如“根据文本”、“好的”）？是否按要求添加了来源标注（如 [1]）？
3. completeness (语义完整度): 预测答案是否包含了标准答案中的所有核心信息点？

请以 JSON 格式输出，包含每个维度的得分和简短的理由。格式如下：
{
    "faithfulness_score": 5,
    "faithfulness_reason": "...",
    "instruction_following_score": 4,
    "instruction_following_reason": "...",
    "completeness_score": 5,
    "completeness_reason": "..."
}"""

async def evaluate_single_case(case_id, data, semaphore):
    """异步评估单个预测结果"""
    async with semaphore:
        user_content = f"""
        【用户问题】: {data['instruction']}
        【参考文本】: {data['input']}
        【标准答案】: {data['ground_truth']}
        
        【预测答案】: {data['prediction']}
        """
        
        try:
            response = await aclient.chat.completions.create(
                model="deepseek-chat", 
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.0 # 评测时需要极其确定的输出
            )
            result = json.loads(response.choices[0].message.content)
            print(f"[✔] 完成测试用例 {case_id} 的评估")
            return result
        except Exception as e:
            print(f"[X] 测试用例 {case_id} 评估失败: {e}")
            return None

async def run_evaluation(test_data, max_concurrent=10):
    """主评测流水线"""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [evaluate_single_case(i, data, semaphore) for i, data in enumerate(test_data)]
    
    print(f"🚀 开始并发评测，共 {len(test_data)} 条测试数据...")
    eval_results = await asyncio.gather(*tasks)
    
    # 统计平均分
    valid_results = [res for res in eval_results if res is not None]
    if not valid_results:
        print("所有评估均失败，请检查 API 配置。")
        return
        
    avg_faithfulness = sum(r['faithfulness_score'] for r in valid_results) / len(valid_results)
    avg_instruction = sum(r['instruction_following_score'] for r in valid_results) / len(valid_results)
    avg_completeness = sum(r['completeness_score'] for r in valid_results) / len(valid_results)
    
    print("\n" + "="*40)
    print("📊 模型微调效果评测报告 (LLM-as-a-Judge)")
    print("="*40)
    print(f"有效评估用例数: {len(valid_results)}")
    print(f"1. 事实一致性 (Faithfulness):       {avg_faithfulness:.2f} / 5.0")
    print(f"2. 指令遵循度 (Instruction Following): {avg_instruction:.2f} / 5.0")
    print(f"3. 语义完整度 (Completeness):        {avg_completeness:.2f} / 5.0")
    print("="*40)
    print("💡 分析建议: 如果指令遵循度低于 4.5，说明模型依然带有闲聊习气，可能需要加大微调的 Epoch 或调整 Prompt。")

# 修改 scripts/llm_judge_eval.py 的 main 函数入口部分
if __name__ == "__main__":
    # 读取批量推理出来的预测结果
    predictions_file = "../data/processed/eval_predictions.jsonl"
    
    test_data = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
            
    # 启动异步高并发评测
    # 注意：并发数取决于你的 DeepSeek API 的并发限制，如果报 HTTP 429 错误，请把 max_concurrent 调小
    asyncio.run(run_evaluation(test_data, max_concurrent=5))