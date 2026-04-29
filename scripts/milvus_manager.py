# milvus_manager.py
import json
import os
from pymilvus import (
    MilvusClient, DataType, FieldSchema, CollectionSchema
)
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
        print(f">>> 连接本地 Milvus 数据库: {db_path}")
        # Milvus Lite 的魔法：直接指定一个本地文件路径即可
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.dim = 1024 # BGE-M3 的输出维度是 1024，如果你用 bge-base 就是 768
        
        # 懒加载 embedding 模型（仅在灌库或查询时才加载，省内存）
        self.embed_model = None

    def create_collection(self):
        """定义表结构 (Schema) - 这是体现 AI Infra 工程素养的关键"""
        if self.client.has_collection(collection_name=self.collection_name):
            print(f"集合 {self.collection_name} 已存在，跳过创建。")
            return

        print(">>> 正在定义集合 Schema ...")
        # 1. 创建字段
        fields = [
            # 主键，自动生成 ID
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # 存储 Chunk 的纯文本，最大长度设大一点防截断
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            # 存储对应的向量
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            # Metadata：比如来源是哪本书，方便后续做条件过滤 (Metadata Filtering)
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255)
        ]
        
        schema = CollectionSchema(fields=fields, description="建筑史知识库")
        
        # 2. 创建集合
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema
        )
        
        # 3. 为向量字段建立索引 (加快检索速度)
        # 即使是本地 Lite 版，也可以使用 HNSW 这种高级图索引
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE", # 我们之前用的余弦相似度
            params={"M": 16, "efConstruction": 200}
        )
        self.client.create_index(self.collection_name, index_params)
        print(">>> 集合与索引创建完毕！")

    def insert_jsonl_data(self, jsonl_path):
        """将你之前的 JSONL 数据灌入数据库"""
        if self.embed_model is None:
            print("加载 Embedding 模型...")
            self.embed_model = SentenceTransformer('BAAI/bge-m3')

        print(">>> 开始解析并插入数据...")
        insert_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data.get("input", "").strip()
                if not text:
                    continue
                
                # 计算向量
                vec = self.embed_model.encode(text).tolist()
                
                # 组装一条记录
                insert_data.append({
                    "text": text,
                    "embedding": vec,
                    "source": "中外建筑史" # 这里可以动态解析你的数据源
                })

        # 批量插入
        res = self.client.insert(
            collection_name=self.collection_name,
            data=insert_data
        )
        print(f"成功插入 {res['insert_count']} 条记录！")

    def search(self, query, top_k=10):
        """向量检索"""
        if self.embed_model is None:
            self.embed_model = SentenceTransformer('BAAI/bge-m3')

        query_vec = self.embed_model.encode(query).tolist()
        
        # 从 Milvus 中搜索最相似的 top_k 个结果
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[query_vec],
            limit=top_k,
            output_fields=["text", "source"], # 告诉数据库把原文一起返回来
            search_params={"metric_type": "COSINE"}
        )
        
        # 解析返回结果
        retrieved_chunks = []
        for hits in search_res:
            for hit in hits:
                retrieved_chunks.append(hit['entity']['text'])
                
        return retrieved_chunks

# ================= 测试入口 =================
if __name__ == "__main__":
    db_manager = VectorDBManager()
    
    # 1. 首次运行：建表
    db_manager.create_collection()
    
    # 2. 首次运行：把你的数据灌进去
    # db_manager.insert_jsonl_data("your_data.jsonl")
    
    # 3. 检索测试
    # results = db_manager.search("沃波尔在草莓山住宅使用了什么风格？", top_k=5)
    # for res in results:
    #     print(res[:50] + "...")