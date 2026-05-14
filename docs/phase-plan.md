# AI 科研助手 (Research Assistant) 项目计划

## 项目阶段

### Phase 1: 项目初始化
项目骨架、依赖安装、基础配置。

### Phase 2: RAG 基础模块
文档 → 分割 → 嵌入 → 存储 → 检索全流程。

### Phase 3: Agent 工具定义
理解 `@tool` 装饰器定义工具。

### Phase 4: ReAct Agent 实现
推理 + 行动循环。`create_react_agent` + `AgentExecutor`。

### Phase 5: 对话记忆
多轮对话上下文 + 文件持久化。

### Phase 6: Web UI
Streamlit 聊天界面。

### Phase 7: 集成测试与优化

---

## Future: 多模态支持
支持论文图片理解（图表、流程图等）。

---

## 详细文档

**经验教训**: `docs/lessons/phase-7.5.1-file-tool-debugging.md`, `phase-7.6-memory-persistence.md`

**完整项目文档**: `README.md`

---

## 验证命令

```bash
# 初始化知识库
python scripts/init_knowledge_base.py

# 运行 Agent 测试
python -m tests.test_research_agent

# 查看/清空记忆
python scripts/manage_memory.py --show-chat
python scripts/manage_memory.py --show-scratchpad
python scripts/manage_memory.py --clear
```