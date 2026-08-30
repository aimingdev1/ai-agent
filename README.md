# AI Agent 智能体服务（ReAct + RAG）

基于 FastAPI + DeepSeek 实现的智能 Agent 系统：大模型自主判断用户意图，调度工具完成任务，并支持私有知识库检索增强生成（RAG）。

## 功能特性

- **ReAct 推理框架**：LLM 决策（Thought）→ 前缀路由（Action）→ 工具执行（Observation），模型自主选择工具并整合结果
- **三类工具调度**：
  - 计算器：算式求值
  - 联网搜索：基于 DuckDuckGo 的实时信息检索
  - **RAG 知识库检索**：私有文档分块 → TF-IDF 向量化 → 余弦相似度检索 Top-3 相关片段
- **检索增强生成**：RAG 分支检索到资料后回灌 LLM，基于给定资料作答（资料中没有则如实说明），有效减少模型幻觉
- **Web 交互界面**：Chat 风格对话页面，开箱即用

## 项目结构

```
agent_api.py    # 全部核心代码：LLM客户端、工具函数、RAG管线、FastAPI路由、前端页面
.env            # API Key 配置（不入库）
```

## 核心流程

```
用户输入
   │
   ▼
agent_think()  ── LLM 判断意图（System Prompt 定义三工具协议）
   │
   ├─ "CALCULATE: ..."  ──► calculator_tool()      ──► 返回计算结果
   ├─ "SEARCH: ..."     ──► search_tool()          ──► 返回搜索结果
   ├─ "RAG: ..."        ──► rag_tool() 检索知识库
   │                          │
   │                          ▼
   │                       rag_answer() 资料回灌 LLM ──► 基于资料生成答案
   └─ 其他                ──► 直接返回 LLM 回复
```

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn openai python-dotenv ddgs
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=你的密钥
```

API Key 可在 [DeepSeek 开放平台](https://platform.deepseek.com/) 申请。

### 3. 启动服务

```bash
python -m uvicorn agent_api:app --reload
```

浏览器打开 `http://127.0.0.1:8000` 即可对话。

### 4. 测试效果

- 输入 `2+3` → 走计算器
- 输入 `今天有什么AI新闻` → 走联网搜索
- 输入 `介绍一下RAG` → 走知识库检索（回答会基于检索到的资料）

## 技术栈

Python · FastAPI · DeepSeek API · TF-IDF 向量检索 · DuckDuckGo Search

## 后续计划

- [ ] 接入真实 Embedding 模型（替代 TF-IDF）提升检索精度
- [ ] 支持上传 PDF/Word 文档自动入库
- [ ] 升级为多圈 ReAct 循环（记忆 + 多步推理）
