import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from utils import clean_and_chunk_docx
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的变量到系统环境变量中
load_dotenv()

# 安全地获取配置，如果没找到则返回 None
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")

if not API_KEY:
    raise ValueError("未找到 API_KEY，请检查 .env 文件配置！")

# 初始化客户端
client = OpenAI(
    api_key=API_KEY, 
    base_url=BASE_URL 
)

def generate_dynamic_qa_pairs(chunk_text):
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
    "instruction": "提出一个迷惑性问题，具体询问的细节（如设计师、年代、具体材料等）在【参考文本】中绝对没有提到，也可以捏造一个细节或问其他时代的建筑，或者与文本核心实体相关但细节未提及（负样本）",
    "input": "提供的参考文本",
    "output": "抱歉，提供的参考资料中未包含关于[问题核心词汇]的相关信息。"
  }
]"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat", # 例如 "deepseek-chat" 或 "qwen-max"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"【参考文本】：\n{chunk_text}"}
            ],
            response_format={"type": "json_object"}, # 强制输出 JSON
            temperature=0.7
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"生成失败: {e}")
        return []


# 主流程演示
def build_dataset(chunks, output_file="rag_sft_dataset.jsonl"):
    with open(output_file, 'w', encoding='utf-8') as f:
        # 使用 tqdm 显示进度条，假设我们先拿前 5 个 chunk 测试
        for chunk in tqdm(chunks[:2]): 
            qa_pairs = generate_dynamic_qa_pairs(chunk)
            for qa in qa_pairs:
                # 写入 JSONL 格式（一行一个 JSON，微调框架标准格式）
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')



# 测试提取
chunks = clean_and_chunk_docx("../data/raw/西方艺术史.docx")
print(f"共提取了 {len(chunks)} 个文本切片")

# 运行合成
build_dataset(chunks)