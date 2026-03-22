# CDS547 小组作业材料说明

本文件夹内 PDF 为课程模板/指南；下列文件为**已按任务填好的可提交版本**（请替换个人信息后使用）。

## 1. 项目提案（对应 `CDS547_Project_Proposal.pdf`）

| 文件 | 说明 |
|------|------|
| **`CDS547_Project_Proposal_Filled.md`** | 已填写题目、目标、预期结果、时间线、参考文献；成员表为占位符，请改成你们小组真实信息。 |

**如何变成 PDF：** 用 Word / Google Docs 打开并导出 PDF，或用 Pandoc：`pandoc CDS547_Project_Proposal_Filled.md -o proposal.pdf`（需安装 Pandoc）。

若课程要求**必须使用原 PDF 表格**，把 `CDS547_Project_Proposal_Filled.md` 里的内容复制进官方 Word/PDF 模板即可。

---

## 2. 进度报告（对应 `Guideline for Group Project Progress Report.pdf`）

指南要求：

- 使用 Overleaf 提供的 LaTeX 模板：<https://www.overleaf.com/read/gkdrydjrystj#436146>
- **篇幅：** 正文 5–8 页（不含附录与参考文献）
- **提交：** 编译后的 PDF  
- **截止：** 2026 年 3 月 11 日 12:00 AM（以课程公告为准）

| 文件 | 说明 |
|------|------|
| **`progress_report/main.tex`** | 符合指南意图的进度报告正文结构：技术进展、初步结果与批判性分析、计划变更、项目管理与时间线、下一步。 |

**推荐做法：**

1. 在 Overleaf 打开课程模板，把 `main.tex` 中各 `\section{...}` 及段落**复制进模板**对应位置；或  
2. 本地安装 TeX Live / MiKTeX，在 `progress_report` 目录执行：  
   `pdflatex main.tex`  
   （多轮编译以更新引用；若改用 `bibtex` 需自行改文献方式。）

**必做修改：** 全文搜索 `[Replace]`，填入真实小组名、成员、数据集、实现细节与实验数字。

---

## 3. 指南强调的评分意图（写作时请覆盖）

- 展示**技术进展**：调研、实现、实验。  
- **批判性思考**：挑战、对早期结果的分析、计划调整。  
- **项目管理**：是否按计划推进、分工是否落实。

---

如有课程提供的最新模板链接或页数要求变更，以 Canvas / 邮件为准。

---

## 4. 技术实现（RAG 工程）

完整可运行代码在 **`rag_qa/`** 目录：`README.md` 含安装、配置、`build` / `query` / `eval` / `ablate-topk` 用法。与提案中的 RAG 课题一致，可直接用于进度报告与期末报告的技术章节与实验。
