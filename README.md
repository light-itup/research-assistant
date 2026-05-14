# AI 科研助手 (Research Assistant)

基于 LangChain + LlamaIndex 的 AI 科研助手，用于学习 RAG + Agent 架构。

## 核心功能

- **RAG 知识库** - 文档检索增强生成
- **网络搜索** - Tavily 实时搜索
- **代码助手** - 代码解释与执行
- **多轮对话** - 记忆上下文

---

## 快速开始

```bash
# 1. 初始化知识库（首次或文档变更后）
python scripts/init_knowledge_base.py

# 2. 运行测试
python -m tests.test_research_agent

# 3. 管理记忆
python scripts/manage_memory.py --show-chat
python scripts/manage_memory.py --show-scratchpad
python scripts/manage_memory.py --clear
```

---

## 项目结构

```
src/
├── agents/research_agent.py      # ReAct Agent 核心
├── tools/                        # 工具 (RAG/搜索/代码/文件)
├── rag/                          # RAG 模块 (加载/分割/向量化/存储)
├── memory/                       # 记忆 (chat history + scratchpad)
└── config/                       # 配置
scripts/
├── init_knowledge_base.py        # 知识库初始化
└── manage_memory.py             # 记忆管理
data/
├── knowledge_base/               # 知识库文档
└── memory/                      # 持久化存储
docs/lessons/                    # 经验教训文档
```

---

## 技术要点

### ReAct 模式

Agent 循环：`Thought → Action → Action Input → Observation → ... → Final Answer`

LangChain 的 `create_react_agent` 创建的管道包含：
1. `prompt` - 模板
2. `llm` - 模型
3. `ReActSingleInputOutputParser` - 正则解析 Action/Action Input

### 全局索引管理器

避免每次查询重新构建索引：
- `IndexManager` 单例模式
- 手动初始化后复用

### 记忆持久化

- `persistent_memory.py` - chat history 持久化到 JSON
- `scratchpad_history.py` - agent 思考过程持久化

---

## 经验教训

详见 `docs/lessons/` 目录：

| 文件 | 内容 |
|------|------|
| `phase-7.5.1-file-tool-debugging.md` | 工具调试问题汇总 |
| `phase-7.6-memory-persistence.md` | 记忆持久化实现 |

---

## 详细文档

- `docs/phase-plan.md` - 项目计划
- `CLAUDE.md` - Claude Code 指导文件