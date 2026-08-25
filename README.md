# 🧭 AI 旅行 Agent(基于 Python + LangChain)

根据简历「基于 AI Agent 的旅行助手」项目经历搭建的完整工程实现:
一句话自然语言 → **LLM 理解 → RAG 检索 → 可行性评估 → 多日行程生成**(含每日景点/餐饮/住宿/预算)。

## 架构

```
用户输入 / 语音识别
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│ LangGraph 工作流 (app/agent/graph.py)                        │
│                                                              │
│ ① parse —— LLM 理解(DeepSeek)                                │
│    输出结构化 JSON:目的地/日期/天数/预算/偏好                   │
│        │ 信息不足 → clarify 澄清追问                           │
│        ▼                                                      │
│ ② retrieve —— RAG 检索                                        │
│    旅行案例知识库 → 语义切片(chunk) → 向量化 → 相似度检索        │
│        │ (app/rag/*)                                         │
│        ▼                                                      │
│ ③ evaluate —— 可行性评估                                      │
│    天气工具(Open-Meteo) + 预算测算 + 知识库季节/人流建议         │
│        │ (app/tools/weather.py, budget.py)                   │
│        ▼                                                      │
│ ④ generate —— 旅行建议生成                                    │
│    LLM 生成 Markdown 行程,可调用高德地图工具                    │
│    (app/tools/amap.py: 地理编码/POI 搜索/周边查询)              │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
UI: CLI (app/ui/cli.py) / Streamlit (app/ui/streamlit_app.py)
```

## 技术栈

| 层 | 组件 |
| --- | --- |
| 编排 | LangGraph(StateGraph 多节点工作流) |
| LLM | LangChain DeepSeek 官方接入(`langchain-deepseek`) |
| RAG | 自研数据清洗 + 中文感知 Chunking + 向量检索(内置轻量库 / ChromaDB 可选) |
| 工具 | 高德地图 REST(地理编码/POI/周边)、Open-Meteo 天气(免费无 Key)、预算测算 |
| UI | 交互式 CLI + Streamlit Web 原型 |
| 降级 | 无 API Key → Mock 模式全流程可演示;LLM 失败 → 知识库模板行程兜底 |

## 快速开始

### 1. 安装

```bash
cd ai-travel-agent
python -m venv .venv
# Windows:
.venv\Scripts\activate
# 或直接:
.venv\Scripts\python -m pip install -r requirements.txt
```

### 2. 配置密钥(可选但推荐)

```bash
copy .env.example .env    # 编辑 .env 填入
```

| 变量 | 说明 | 获取方式 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 大模型调用(不填则演示模式) | https://platform.deepseek.com |
| `AMAP_KEY` | 高德地图 Web 服务 Key(不填则 POI 工具降级) | https://console.amap.com |
| `EMBEDDING_PROVIDER` | `auto` / `sentence-transformers`(推荐本地 bge-small-zh-v1.5)/ `local_hash` | 默认 auto |

### 3. 构建知识库(首次自动,也可手动)

```bash
.venv\Scripts\python scripts\build_kb.py
```

### 4. 运行

```bash
# CLI 交互模式
.venv\Scripts\python -m app.ui.cli

# 单次示例
.venv\Scripts\python -m app.ui.cli --demo "国庆想去西安玩4天,人均3000,喜欢历史和美食"

# Streamlit Web 对话框(聊天式)
.venv\Scripts\streamlit run app/ui/streamlit_app.py
```

Web 对话框功能:
- 聊天气泡对话,行程 Markdown 渲染
- **对话上下文记忆**:跨轮记住目的地/日期/预算/偏好,省略说法也能理解(如「预算降到 2500」「把目的地换成成都」)
- **多轮对话修订**:获得行程后可直接说「预算降到 2000」「第 2 天轻松一点」,Agent 会基于上一版行程生成新版(配置 DeepSeek Key 后为智能修订)
- 每次回答可展开「🔍 处理过程」:需求解析 → RAG 检索(含知识库片段)→ 可行性评估 → 行程生成
- 侧边栏:快捷表单一键生成、知识库状态、新建对话

## 项目结构

```
ai-travel-agent/
├── config.py                 # 全局配置(.env)
├── requirements.txt
├── .env.example
├── data/
│   └── travel_cases.json     # RAG 知识库:7 城市旅行案例(JSON)
├── scripts/
│   └── build_kb.py           # 知识库构建:清洗→切片→向量化→入库→自检
├── app/
│   ├── llm.py                # LLM 接入(DeepSeek/Mock)+ 结构化输出解析
│   ├── prompts.py            # Prompt 设计(解析/评估/生成/澄清)
│   ├── rag/
│   │   ├── loader.py         # JSON 数据清洗归一化
│   │   ├── chunker.py        # 中文感知切片(语义级预切 + 二次切分)
│   │   ├── embeddings.py     # 向量化 provider(bge / dashscope / local_hash)
│   │   ├── vector_store.py   # 向量库(内置轻量库 / ChromaDB)
│   │   └── retriever.py      # 检索:城市软过滤 + 阈值 + 上下文格式化
│   ├── tools/
│   │   ├── amap.py           # 高德:地理编码/POI 搜索/周边查询(工具)
│   │   ├── weather.py        # Open-Meteo 天气预报(工具)
│   │   └── budget.py         # 预算测算
│   ├── agent/
│   │   ├── state.py          # LangGraph 状态
│   │   ├── nodes.py          # 四个流程节点 + 澄清节点
│   │   ├── graph.py          # 工作流编排(编译 + 运行)
│   │   └── fallback_plan.py  # 模板行程兜底
│   └── ui/
│       ├── cli.py            # 命令行入口
│       └── streamlit_app.py  # 网页对话框:聊天气泡 + 多轮修订 + 处理过程可视化
└── docs/
    ├── PRD.md                # 产品需求文档
    └── prompt_design.md      # Prompt 设计文档
```

## 与简历项目经历的对应关系

| 简历经历 | 本仓库实现 |
| --- | --- |
| 接入 DeepSeek API 智能问答 | `app/llm.py`(langchain-deepseek) |
| 构建 RAG 知识库 + 向量数据库 | `app/rag/*` + `scripts/build_kb.py` |
| 旅行案例 JSON 数据清洗格式化 | `app/rag/loader.py` |
| Chunking 切片优化 / Embedding 模型更换 | `app/rag/chunker.py` / `embeddings.py`(3 种 provider 可切换) |
| MCP 协议调用高德地图选点 | `app/tools/amap.py`(function calling,与 MCP 语义等价) |
| 用户输入→LLM 理解→RAG 检索→可行性评估→建议生成 | `app/agent/graph.py`(LangGraph 逐节点对应) |
| 原型本地部署与功能测试 | CLI + Streamlit + `scripts/build_kb.py` 自检 |
| PRD / 业务流程图 / Prompt 设计文档 | `docs/PRD.md`、`docs/prompt_design.md` |

## 扩展方向

- **MCP 接入**:将 `app/tools/amap.py` 封装为 MCP Server,供任意 MCP 客户端调用
- **语音输入/输出**:接入 ASR/TTS,复用现有 `parse_node` 的文本化入口
- **图片识别**:上传攻略截图 → 多模态模型提取景点 → 写入知识库
- **多轮对话**:LangGraph 增加 `ChatHistory` 与「行程修订」子图
- **更大知识库**:切换 ChromaDB / Milvus,扩充城市与案例(数据格式保持 `travel_cases.json` Schema)
