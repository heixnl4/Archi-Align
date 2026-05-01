import os
import sys
# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils import clean_and_chunk_docx
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
    system_prompt = """你是一个严谨的艺术史学术数据提取引擎。
你的任务是提取【参考文本】中的核心艺术史知识，动态生成用于训练 RAG 系统的问答对。

【提取与生成规则】（严格遵守）：
1. 问题（instruction）中绝对不能出现“西方艺术史课程”、“根据参考文本”、“文中提到”、“课程中”、“教材指出”等字眼。必须直接提问核心知识点。
2. 动态数量：评估信息密度，根据有效知识点数量，生成对应数量的正样本。知识点越密集，生成的正样本越多（数量在 2-5 个左右）
3. 负样本构造：针对文本的核心实体，提出一个具体的、但在文本中未提及细节的迷惑性问题（同样不允许使用“参考文本”等前缀词）。

【输出 JSON 格式要求】（必须是包含多个对象的数组）：
[
  {
    "instruction": "根据参考文本提出的具体问题（正样本）",
    "input": "提供的参考文本",
    "output": "详细且严谨的回答，并使用 [1] 标注信息来源。"
  },
  // ... (根据信息密度，动态添加更多的正样本 JSON 对象)
  {
    "instruction": "提出一个迷惑性问题，具体询问的细节（如人物、年代等）在【参考文本】中绝对没有提到，也可以捏造一个细节或问其他时代的艺术作品，或者与文本核心实体相关但细节未提及（负样本）",
    "input": "提供的参考文本",
    "output": "抱歉，提供的参考资料中未包含关于[问题核心词汇]的相关信息。"
  }
]"""

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

async def build_dataset_async(chunks, output_file="../../data/processed/rag_sft_dataset.jsonl", max_concurrent=10):
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
    chunks = clean_and_chunk_docx("../../data/raw/西方艺术史.docx")
    print(f"共提取了 {len(chunks)} 个文本切片")
    
    # Python 3.7+ 运行异步主函数的标准做法
    asyncio.run(build_dataset_async(chunks, max_concurrent=50))