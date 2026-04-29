# qdrant_manager.py
import json
import uuid
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

# ================= 路径动态推导 =================
# 1. 获取当前脚本 (qdrant_manager.py) 所在目录的绝对路径 (.../Archi-Align/scripts)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 推导出项目根目录 (.../Archi-Align)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# 3. 定义数据库存放位置 (.../Archi-Align/data/qdrant_db)
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "qdrant_db")
# ================================================

class VectorDBManager:
    def __init__(self, db_path=DEFAULT_DB_PATH, collection_name="arch_history"):
        print(f">>> 连接本地 Qdrant 数据库 (Windows 原生支持): {db_path}")
        # Qdrant 本地模式：数据会直接保存在你指定的文件夹中
        # 确保 data 文件夹和 qdrant_db 文件夹存在，如果不存在则自动创建
        os.makedirs(db_path, exist_ok=True)
        
        print(f">>> 连接本地 Qdrant 数据库: {db_path}")
        self.client = QdrantClient(path=db_path) 
        self.collection_name = collection_name
        self.dim = 1024 
        self.embed_model = None
        self.model_path = "D:/develop/models/bge-m3"

    def create_collection(self):
        """定义表结构 (Schema)"""
        # 检查集合是否存在
        if self.client.collection_exists(collection_name=self.collection_name):
            print(f"集合 {self.collection_name} 已存在，跳过创建。")
            return

        print(">>> 正在定义集合 Schema ...")
        # 创建集合，指定向量维度和距离度量公式（余弦相似度）
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        print(">>> 集合创建完毕！")

    def insert_jsonl_data(self, jsonl_path):
        """将 JSONL 数据灌入 Qdrant (修复了重复插入问题)"""
        if self.embed_model is None:
            print(f"加载 Embedding 模型: {self.model_path}")
            self.embed_model = SentenceTransformer(self.model_path, device='cpu')

        print(">>> 开始解析并插入数据...")
        points = []
        
        if not os.path.exists(jsonl_path):
            print(f"错误: 找不到文件 {jsonl_path}")
            return

        # 使用一个 Set 在内存里先做一层去重，防止同一个 JSONL 里就有重复行
        seen_texts = set()

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("input", "").strip()
                    
                    if not text or text in seen_texts:
                        continue
                    seen_texts.add(text)
                    
                    vec = self.embed_model.encode(text).tolist()
                    
                    # 【核心修改】: 使用 uuid5 基于文本内容生成确定性 ID
                    # NAMESPACE_DNS 是一个标准盐值，配合你的文本生成固定的 UUID
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text))
                    
                    points.append(
                        PointStruct(
                            id=point_id, 
                            vector=vec, 
                            payload={"text": text, "source": "中外建筑史"}
                        )
                    )
                except Exception as e:
                    print(f"解析行出错: {e}")

        if points:
            # upsert 的机制：如果 point_id 已经存在，则覆盖；不存在，则新增
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"成功插入或更新了 {len(points)} 条去重后的记录！")

    def search(self, query, top_k=10):
        """向量检索"""
        if self.embed_model is None:
            self.embed_model = SentenceTransformer(self.model_path, device='cpu')

        query_vec = self.embed_model.encode(query).tolist()
        
        # 执行检索   
        # 推荐使用 query_points，它是 search 的升级版，在本地模式下更稳定
        search_res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
            with_payload=True  # 必须开启，不然只返回 ID 不返回文本
        ).points

        # 提取原文
        retrieved_chunks = [hit.payload["text"] for hit in search_res]
        return retrieved_chunks

# ================= 测试入口 =================
if __name__ == "__main__":
    db_manager = VectorDBManager()
    db_manager.create_collection()
    
    # 取消注释以下代码进行首次数据灌注
    # db_manager.insert_jsonl_data("../data/processed/test_chunks.jsonl")

    # 3. 检索测试
    results = db_manager.search("沃波尔在草莓山住宅使用了什么风格？", top_k=5)
    for res in results:
        print(res[:50] + "...")