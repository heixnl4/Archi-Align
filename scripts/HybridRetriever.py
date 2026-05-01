import os
# 将模型统一下载到你当前项目下的 model_cache 文件夹中
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "D:/develop/model"
import json
import jieba
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util

class HybridRetriever:
    def __init__(self, jsonl_path):
        print(">>> 1. 正在初始化检索器与模型...")
        # 1. 加载数据
        self.chunks = self._load_data(jsonl_path)
        
        # 2. 初始化 BM25 (稀疏检索)
        # 对所有 chunk 进行中文分词
        self.tokenized_corpus = [list(jieba.cut(chunk)) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 3. 初始化 BGE 向量模型 (稠密检索)
        # 自动使用 GPU (如果本地有的话)，否则用 CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dense_model = SentenceTransformer('D:/develop/models/bge-m3', device=self.device)
        print("   - 正在计算全量知识库的向量(Embedding)...")
        self.chunk_embeddings = self.dense_model.encode(self.chunks, convert_to_tensor=True)
        
        # 4. 初始化 BGE Reranker (重排序)
        self.reranker = CrossEncoder('D:/develop/models/bge-reranker-base', device=self.device)
        print(">>> 初始化完成！\n")

    def _load_data(self, path):
        """从你的 JSONL 文件中提取所有的 chunk (input字段)"""
        chunks = []
        # 这里假设你的 jsonl 格式为 {"input": "具体的文本内容", ...}
        # 如果你只提取了纯文本切片，按需修改即可
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if "input" in data and data["input"].strip():
                    chunks.append(data["input"])
        # 去重，防止相同的 chunk 干扰
        return list(set(chunks))

    def retrieve(self, query, top_k_recall=10, top_k_rerank=3):
        """
        完整的检索流水线：双路召回 -> 去重合并 -> 精排
        """
        print(f"【用户提问】: {query}")
        
        # ==========================================
        # 阶段一：双路并行召回 (Recall)
        # ==========================================
        # 1.1 BM25 召回
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 获取得分最高的 Top-K 索引
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k_recall]
        
        # 1.2 向量召回
        query_embedding = self.dense_model.encode(query, convert_to_tensor=True)
        # 计算余弦相似度
        cos_scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
        # 获取得分最高的 Top-K 索引
        dense_top_idx = torch.topk(cos_scores, k=top_k_recall).indices.cpu().numpy()

        # ==========================================
        # 阶段二：结果去重合并 (Merge)
        # ==========================================
        # 使用 set 进行去重，合并两路召回的索引
        merged_indices = list(set(bm25_top_idx).union(set(dense_top_idx)))
        candidate_chunks = [self.chunks[i] for i in merged_indices]
        print(f" -> 双路召回合并后，共提取出 {len(candidate_chunks)} 个候选切片。")

        # ==========================================
        # 阶段三：Cross-Encoder 重排序 (Rerank)
        # ==========================================
        # 构造给 Reranker 的输入格式: [(query, chunk1), (query, chunk2), ...]
        rerank_pairs = [[query, chunk] for chunk in candidate_chunks]
        # 进行精细打分
        rerank_scores = self.reranker.predict(rerank_pairs)
        
        # 将切片和它的精排得分绑定，并按得分降序排序
        ranked_results = sorted(zip(candidate_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
        
        # 取最终的 Top-K 喂给大模型
        final_top_chunks = ranked_results[:top_k_rerank]
        
        print(f"\n【最终 Rerank 筛选出的 Top-{top_k_rerank} 切片】:")
        for i, (chunk, score) in enumerate(final_top_chunks):
            print(f"--- 排名 {i+1} (精排得分: {score:.4f}) ---")
            # 截断打印，方便在控制台查看
            print(f"{chunk[:100]}...\n")
            
        return [chunk for chunk, score in final_top_chunks]

# ================= 测试入口 =================
if __name__ == "__main__":
    # 替换为你本地的 jsonl 文件路径
    # 第一次运行会自动从 HuggingFace/ModelScope 下载 bge 模型，需要稍等片刻
    retriever = HybridRetriever("../data/processed/test_chunks.jsonl")
    
    # 用你之前的那个“陷阱”问题来测试一下检索效果
    test_query = "沃波尔在草莓山住宅中使用了哪些具体的哥特式装饰元素？"
    
    # 拿到最终精准的 Context，之后就可以把它拼接到你 SFT 模型的 prompt 里了
    final_context_list = retriever.retrieve(test_query)