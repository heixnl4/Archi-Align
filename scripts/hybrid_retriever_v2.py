# hybrid_retriever_v2.py
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# 导入我们刚才写的 Milvus 管理器
# 假设它保存在 milvus_manager.py 中
from qdrant_manager import VectorDBManager

class HybridRetrieverV2:
    def __init__(self, db_manager: VectorDBManager, all_chunks: list):
        print(">>> 1. 正在初始化混合检索器 (V2 数据库版)...")
        
        # 1. 挂载 Milvus 数据库实例
        self.db = db_manager
        
        # 2. 保存全量文本 (供 BM25 和 精排使用)
        self.chunks = all_chunks
        
        # 3. 初始化 BM25 (稀疏检索，依然在内存中做词频统计)
        print("   - 正在构建 BM25 词频索引...")
        self.tokenized_corpus = [list(jieba.cut(chunk)) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 这里删除了 BGE-M3 的本地加载和全量 embedding 计算
        # 已经交给了 self.db 去持久化管理了。
        
        # 4. 初始化 BGE Reranker (重排序依然需要)
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = CrossEncoder('D:/develop/models/bge-reranker-base', device=self.device)
        print(">>> 检索器 V2 初始化完成！\n")

    def retrieve(self, query, top_k_recall=10, top_k_rerank=3):
        print(f"【用户提问】: {query}")
        
        # ==========================================
        # 阶段一：双路并行召回
        # ==========================================
        
        # 1.1 BM25 召回 (获取文本)
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 获取 Top-K 的索引，然后转成具体文本
        import numpy as np
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k_recall]
        bm25_candidates = [self.chunks[i] for i in bm25_top_idx]
        
        # 1.2 Milvus 向量召回 (直接向数据库要文本)
        # 极其轻量，不再做任何高维矩阵运算
        dense_candidates = self.db.search(query, top_k=top_k_recall)

        # ==========================================
        # 阶段二：结果去重合并
        # ==========================================
        # 因为现在两路返回的都是字符串文本，直接用 set 对文本进行去重
        merged_candidates = list(set(bm25_candidates + dense_candidates))
        print(f" -> BM25 与 Milvus 召回合并后，共提取出 {len(merged_candidates)} 个候选切片。")

        # ==========================================
        # 阶段三：Cross-Encoder 精排
        # ==========================================
        if not merged_candidates:
            return []

        rerank_pairs = [[query, chunk] for chunk in merged_candidates]
        rerank_scores = self.reranker.predict(rerank_pairs)
        
        ranked_results = sorted(zip(merged_candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        final_top_chunks = ranked_results[:top_k_rerank]
        
        print(f"\n【最终 Rerank 筛选出的 Top-{top_k_rerank} 切片】:")
        for i, (chunk, score) in enumerate(final_top_chunks):
            print(f"--- 排名 {i+1} (精排得分: {score:.4f}) ---")
            print(f"{chunk[:100]}...\n")
            
        return [chunk for chunk, score in final_top_chunks]

# ================= 测试入口 =================
if __name__ == "__main__":
    import json
    
    # 1. 准备数据源
    jsonl_path = "../data/processed/test_chunks.jsonl"
    all_chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 【修复关键】：统一提取出来并执行 strip()
            text = data.get("input", "")
            if text.strip():
                # 确保进入 all_chunks 的也是干干净净的字符串
                all_chunks.append(text.strip()) 
                
    # 在内存中对原始文本做一遍绝对一致的去重
    all_chunks = list(set(all_chunks))
    
    # 2. 实例化数据库 (确保你之前已经运行过 create_collection 和 insert_jsonl_data)
    db = VectorDBManager()
    
    # 3. 实例化 V2 版检索器
    retriever = HybridRetrieverV2(db_manager=db, all_chunks=all_chunks)
    
    # 4. 执行检索测试
    results = retriever.retrieve("沃波尔在草莓山住宅中使用了哪些具体的哥特式装饰元素？")