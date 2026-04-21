import os
import json
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")

# 使用异步客户端
aclient = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

async def generate_single_chunk_async(chunk_text, chunk_index, semaphore):
    """
    异步处理单个切片，受 semaphore 控制并发量
    """
    system_prompt = """你是一个严谨的建筑史学术数据提取引擎。
    评估参考文本信息密度，动态生成 RAG 问答对 JSON 数组。
    (此处省略具体的 Prompt，直接复用我们之前讨论过的动态生成 Prompt 即可)"""

    # 信号量：控制同时处于请求状态的协程数量
    async with semaphore:
        print(f"--> 开始处理 Chunk {chunk_index}")
        try:
            response = await aclient.chat.completions.create(
                model="deepseek-chat", # 请替换为你实际使用的模型名
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"【参考文本】：\n{chunk_text}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            result = json.loads(response.choices[0].message.content)
            print(f"<-- 完成 Chunk {chunk_index}，生成了 {len(result)} 条数据")
            return result
        except Exception as e:
            print(f"[X] Chunk {chunk_index} 处理失败: {e}")
            return []

async def build_dataset_async(chunks, output_file="rag_sft_dataset.jsonl", max_concurrent=10):
    """
    异步并发构建数据集
    """
    # 限制最大并发数为 10（可根据 API 限制调整）
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # 创建所有的异步任务
    tasks = [
        generate_single_chunk_async(chunk, i, semaphore) 
        for i, chunk in enumerate(chunks)
    ]
    
    # asyncio.gather 会并发执行所有任务，并按原顺序返回结果
    print(f"启动并发生成，共 {len(chunks)} 个任务，最大并发数: {max_concurrent}...")
    results = await asyncio.gather(*tasks)
    
    # 将结果写入文件
    valid_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa_list in results:
            if not qa_list: continue
            for qa in qa_list:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
                valid_count += 1
                
    print(f"\n🎉 数据生成完毕！成功写入 {valid_count} 条高质量 QA 对。")

# 运行入口
if __name__ == "__main__":
    # 假设你已经有了 chunks 列表
    # chunks = clean_and_chunk_docx("../data/raw/建筑史资料.docx")
    
    # 为了测试，我们用 5 个虚拟切片
    test_chunks = ["测试切片1", "测试切片2", "测试切片3", "测试切片4", "测试切片5"]
    
    # Python 3.7+ 运行异步主函数的标准做法
    asyncio.run(build_dataset_async(test_chunks, max_concurrent=3))