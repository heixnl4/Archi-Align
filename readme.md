
---

# ArchRAG-Qwen-Align: 面向垂直领域（西方建筑艺术史）的 RAG 模型微调与对齐框架

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Model](https://img.shields.io/badge/Model-Qwen2.5--7B-green)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📖 项目简介 (Overview)

本项目旨在将通用大语言模型（Qwen2.5-7B）通过指令微调（SFT）和强化学习对齐（GRPO/RLHF），打造为具备专业学术素养的“垂直领域专家”。

项目核心聚焦于 **RAG（检索增强生成）场景下的痛点**，通过系统性的数据工程、模型微调与自动化评估，解决模型在处理专业长文本时容易出现的“幻觉”、“拒答能力弱”以及“无法严格遵循引用规范格式（如 `[1][2]`）”等问题。本项目不仅仅是一个算法实现，更包含从数据准备到推理部署的完整 AI 系统闭环。

## ✨ 核心特性 (Key Features)

* **⚡ 异步高效的数据处理流**：实现了针对长文档的智能切块与异步 API 调用，高效生成面向 RAG 的 QA 对，并创新性地**构造正负样本**，提升模型的抗干扰能力。
* **🧠 面向业务场景的 SFT 微调**：基于 Qwen2.5-7B，定制化微调模型的专业语气（如建筑学者的学术严谨性）及格式遵循能力（严格的来源引用标注）。
* **📏 LLM-as-a-Judge 自动化评估**：集成 AI 打分模块，摆脱单一的主观盲测，对微调前后的模型进行多维度的量化指标对比（忠实度、指令遵循度）。
* **🛠️ 全栈 AI Infra 视野 (WIP)**：规划包含高性能向量检索、重排序（Rerank）策略、vLLM 推理加速及模型量化部署。

## ⚙️ 训练配置与资源消耗 (Training Config)

为了保证训练的稳定性与硬件资源的有效利用，SFT 阶段经过调优后的核心参数与资源消耗记录如下：

* **Base Model**: Qwen2.5-7B
* **Hardware**: RTX4090 24GB
* **VRAM Usage**: 约 20 GB
* **Hyperparameters**: 
    * `Batch Size`: 2
    * `Gradient Accumulation Steps`: 4
    * `Learning Rate`: 2e-4
    * `Num Train Epochs`: 3

## 🗺️ 项目路线图 (Roadmap)

本项目分为四个核心阶段推进，目前已完成第一阶段的闭环，正向更高阶的 RL 对齐与系统部署进发。

### ✅ Phase 1: 领域专家 SFT 与效果基线建立 (已完成)
- [x] 开发异步文档解析与 RAG QA 对构建脚本（含正负样本构建）。
- [x] 完成基于 Qwen2.5-7B 的 SFT 训练链路。
- [x] 编写模型微调前后的推理对比测试脚本。
- [x] 实现 LLM-as-a-Judge，对 SFT 模型进行定性（Case Study）与定量（拒答率、格式正确率）的“体检”。

### ✅ Phase 2: 构建高精度 RAG 检索闭环 (已完成)
- [x] **Qdrant 本地向量数据库集成**：采用 Qdrant 本地模式持久化存储文档向量与原始文本，替代全量内存加载，实现轻量级、可复用的向量检索底座。集合采用 COSINE 距离度量，支持基于内容哈希的幂等性去重插入（`upsert`）。
- [x] **Embedding 优化**：接入 `BAAI/bge-m3` 专业文本表征模型（1024 维），在灌库与查询阶段进行高质量的语义向量化，显著优于通用基座 Embedding。
- [x] **混合检索与 Rerank 精排**：构建 `HybridRetrieverV2`，融合 **BM25 稀疏检索**（`jieba` 中文分词 + `rank_bm25`）与 **Qdrant 稠密向量检索**双路召回，经去重合并后交由 `bge-reranker-base`（CrossEncoder）进行 Top-K 二次精排，大幅提升垂直概念（如：斗拱、哥特式尖券）的检索命中率与上下文忠实度。
- [x] **FastAPI 检索服务化**：封装为生产级异步服务（`api/server.py`），提供 `/api/retrieve` 标准接口。通过 `lifespan` 生命周期管理全局检索器与模型实例，避免每次请求重复加载，实现低延迟、高可用的 RAG 上下文检索能力。

### 🚀 Phase 3: GRPO/RLHF 强化学习对齐 (规划中)
- [ ] **设计 RAG 专用奖励函数 (Reward Model)**：
    * *忠实度奖励 (Faithfulness)*：模型回答完全基于 Context 且无外部幻觉，给予正向奖励。
    * *拒答奖励 (Rejection)*：Context 缺失关键信息时，模型能果断拒绝回答，给予正向奖励。
    * *格式奖励 (Formatting)*：严格输出准确的 `[1][2]` 引用来源格式。
- [ ] 使用 GRPO 算法进行强化学习微调，彻底纠正模型“自作聪明”的问题。

### 🏗️ Phase 4: 高性能推理与系统部署 (规划中)
- [ ] **vLLM 加速部署**：剥离原生 transformers 慢速推理，采用 vLLM 构建高并发服务接口。
- [ ] **模型量化**：测试 INT8/FP8 量化方案，输出精度损耗与推理速度的对比压测报告。


## 👨‍💻 作者 (Author)

* **Shi Yu** * Focus: LLM Training / Agentic RAG 

---
*If you find this project helpful, please give it a ⭐!*