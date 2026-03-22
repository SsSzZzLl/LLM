# 🤝 团队协作指南 — CDS547 Group Project

## 一、首次设置（每人做一次）

### 1. 克隆仓库
```bash
git clone https://github.com/<你的用户名>/LLM_Group.git
cd LLM_Group
```

### 2. 配置 Python 环境
```bash
cd rag_qa
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 3. 配置 API Key
```bash
cp rag_qa/.env.example rag_qa/.env
# 编辑 .env 填入 OPENAI_API_KEY（不要提交到 git！）
```

---

## 二、日常开发流程

### 📌 分支策略（Feature Branch Workflow）

```
main                  ← 稳定版本，只通过 PR 合并
├── feature/agents    ← 宋卓立: Agent 架构 & 编排
├── feature/data      ← 谢宇轩: 数据处理 & chunking
├── feature/index     ← 陶泽佳: 索引 & 检索优化
├── feature/route     ← 耿源: Route Agent
├── feature/retrieve  ← 薛宇恒: Retrieval Agent
├── feature/reason    ← 叶子然: Reasoning Agent
└── feature/synth     ← 冯嘉颖: Synthesis Agent & 评测
```

### 每次开发前
```bash
# 1. 切换到自己的分支
git checkout feature/<你的分支名>

# 2. 先拉取最新的 main
git pull origin main

# 3. 合并 main 到你的分支（保持同步）
git merge main
# 如果有冲突，解决后 git add + git commit
```

### 写完代码后
```bash
# 1. 添加你改的文件
git add <你改的文件>
# 或者添加所有修改
git add .

# 2. 提交（写清楚做了什么）
git commit -m "feat(route): 实现问题复杂度分类"

# 3. 推送到远程
git push origin feature/<你的分支名>

# 4. 在 GitHub 上创建 Pull Request → main
#    让队长或其他组员 review 后合并
```

---

## 三、Commit 信息规范

格式：`<类型>(<模块>): <描述>`

| 类型 | 用途 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat(route): 实现 LLM 问题分类` |
| `fix` | 修 bug | `fix(retrieve): 修复 hybrid 权重计算` |
| `docs` | 文档 | `docs(report): 添加 evaluation 章节` |
| `test` | 测试 | `test(metrics): 添加 token-F1 单元测试` |
| `refactor` | 重构 | `refactor(pipeline): 拆分为多 Agent 架构` |
| `data` | 数据 | `data(corpus): 添加 HotpotQA 子集` |

---

## 四、文件归属（谁改哪些文件）

| 组员 | 主要负责文件 | 可协作文件 |
|------|-------------|-----------|
| 宋卓立 | `agents/base_agent.py`, `pipeline.py` | `cli.py`, `config.yaml` |
| 谢宇轩 | `ingest.py`, `chunking.py` | `data/corpus/` |
| 陶泽佳 | `index_store.py`, `retrieve.py` | `config.yaml` |
| 耿源 | `agents/route_agent.py` | `prompts.py` |
| 薛宇恒 | `agents/retrieval_agent.py` | `retrieve.py` |
| 叶子然 | `agents/reasoning_agent.py` | `prompts.py` |
| 冯嘉颖 | `agents/synthesis_agent.py`, `metrics.py` | `eval/` |

> ⚠️ 如果你需要改别人负责的文件，先在群里说一声，避免冲突！

---

## 五、避免冲突的建议

1. **经常 pull**：每次开始写代码前先 `git pull origin main`
2. **小步提交**：不要攒一大堆代码一起提交
3. **不要改 `.env`**：API key 各自本地配置
4. **不要提交 `data/index/`**：index 是生成的，太大了
5. **PR 前运行测试**：确保不会破坏别人的代码
