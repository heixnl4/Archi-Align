import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import docx
import re
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

def generate_qa_pairs(chunk_text):
    system_prompt = """你是一个专业的建筑史学术数据生成器。
你的任务是根据提供的【参考文本】，生成用于训练 RAG（检索增强生成）系统的问答对。
你必须严格输出一个 JSON 格式的数组，包含两个问答对：一个正样本，一个负样本。

【输出 JSON 格式要求】：
[
  {
    "instruction": "根据参考文本提出的具体问题",
    "input": "提供的参考文本",
    "output": "详细且严谨的回答，并使用 [1] 标注信息来源。"
  },
  {
    "instruction": "提出一个具有高度迷惑性的问题。这个问题必须与【参考文本】中的某个核心实体（如建筑名、人物）相关，但具体询问的细节（如设计师、年代、具体材料等）在【参考文本】中绝对没有提到，也可以捏造一个细节或问其他时代的建筑。",
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
            qa_pairs = generate_qa_pairs(chunk)
            for qa in qa_pairs:
                # 写入 JSONL 格式（一行一个 JSON，微调框架标准格式）
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')

def clean_and_chunk_docx(file_path, chunk_size=500, overlap=50):
    """
    读取 Word 文档，清洗无用字符，并按字数进行滑动窗口切片
    """
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        text = para.text.strip()
        # 简单清洗：去除连续的空格和特殊符号
        text = re.sub(r'\s+', ' ', text)
        if len(text) > 5:  # 过滤掉太短的无意义段落
            full_text.append(text)
            
    content = " ".join(full_text)
    
    chunks = []
    # 滑动窗口切片，overlap 保证上下文不被生硬截断
    for i in range(0, len(content), chunk_size - overlap):
        chunk = content[i:i + chunk_size]
        if len(chunk) > 100: # 过滤掉末尾太短的切片
            chunks.append(chunk)
            
    return chunks

# 测试提取
chunks = clean_and_chunk_docx("../dataset/raw/外国建筑史.docx")
print(f"共提取了 {len(chunks)} 个文本切片")

# 运行合成
build_dataset(chunks)