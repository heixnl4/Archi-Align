# server.py
import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 导入咱们写好的底层模块
from src.retrieval.qdrant_manager import VectorDBManager
from src.retrieval.hybrid_retriever import HybridRetrieverV2

# ================= 1. Pydantic 数据模型 =================
class QueryRequest(BaseModel):
    query: str
    top_k_recall: int = 10
    top_k_rerank: int = 3

class QueryResponse(BaseModel):
    query: str
    contexts: list[str]

# ================= 2. 生命周期管理 (硬核显存控制) =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*50)
    print("🚀 正在启动 RAG 检索服务...")
    print("="*50)
    
    # 1. 动态获取数据文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    jsonl_path = os.path.join(project_root, "data", "processed", "test_chunks.jsonl")
    
    # 2. 读取并清洗全量文本 (供 BM25 使用)
    all_chunks = []
    if not os.path.exists(jsonl_path):
        raise RuntimeError(f"找不到数据文件: {jsonl_path}")
        
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get("input", "")
            if text.strip():
                all_chunks.append(text.strip())
    all_chunks = list(set(all_chunks))
    print(f"✅ 成功加载并去重 {len(all_chunks)} 条纯文本用于 BM25 索引。")
    
    # 3. 实例化全局组件
    print("⏳ 正在挂载 Qdrant 数据库与 BGE 模型...")
    db_manager = VectorDBManager()
    
    print("⏳ 正在初始化混合检索器与 Reranker 模型...")
    retriever = HybridRetrieverV2(db_manager=db_manager, all_chunks=all_chunks)
    
    # 将实例化后的 retriever 挂载到 app 的全局状态中
    app.state.retriever = retriever
    print("\n✅ 服务启动完成，模型已驻留内存/显存！准备接收请求。")
    yield
    
    # 这里可以写关闭服务时的清理逻辑（本例中暂不需要）
    print("🛑 服务正在关闭，释放资源...")

# ================= 3. 实例化 FastAPI =================
app = FastAPI(title="建筑史 RAG 检索服务", lifespan=lifespan)

# ================= 4. 定义 API 路由 =================
@app.post("/api/retrieve", response_model=QueryResponse)
async def retrieve_contexts(request: QueryRequest):
    try:
        # 直接调用全局状态里已经加载好的 retriever
        results = app.state.retriever.retrieve(
            query=request.query,
            top_k_recall=request.top_k_recall,
            top_k_rerank=request.top_k_rerank
        )
        return QueryResponse(query=request.query, contexts=results)
    except Exception as e:
        # 如果内部出错，返回 500 状态码和具体错误
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 启动服务器，host="0.0.0.0" 允许局域网内其他设备访问，端口为 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)