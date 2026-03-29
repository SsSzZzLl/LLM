"""
Chunking 策略 Ablation 实验框架
对比不同分块策略的效果
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np

from rag_qa.chunking_strategies import (
    chunk_document, 
    CHUNKING_STRATEGIES,
    Chunk
)


@dataclass
class ChunkingMetrics:
    """分块策略评估指标"""
    strategy: str
    num_chunks: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    std_chunk_size: float
    total_chars: int
    processing_time: float
    avg_sentence_count: float
    overlap_ratio: float


@dataclass
class AblationResult:
    """单个实验结果"""
    document_name: str
    document_length: int
    metrics: ChunkingMetrics
    sample_chunks: List[str]  # 前3个chunk的预览


class ChunkingAblationExperiment:
    """Chunking策略Ablation实验"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("ablation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # 定义要测试的策略和参数
        self.strategies_config = {
            "sliding_window": {
                "max_chars": 1000,
                "overlap_chars": 150,
            },
            "token_level": {
                "max_tokens": 500,
                "overlap_tokens": 50,
            },
            "semantic": {
                "max_sentences": 10,
                "similarity_threshold": 0.7,
            },
            "langchain_recursive": {
                "chunk_size": 1000,
                "chunk_overlap": 150,
            },
            "paragraph_based": {
                "max_chars": 1000,
                "overlap_chars": 150,
            },
        }
    
    def calculate_metrics(
        self,
        chunks: List[Chunk],
        strategy: str,
        processing_time: float
    ) -> ChunkingMetrics:
        """计算分块指标"""
        if not chunks:
            return ChunkingMetrics(
                strategy=strategy,
                num_chunks=0,
                avg_chunk_size=0,
                min_chunk_size=0,
                max_chunk_size=0,
                std_chunk_size=0,
                total_chars=0,
                processing_time=processing_time,
                avg_sentence_count=0,
                overlap_ratio=0
            )
        
        chunk_sizes = [len(c.text) for c in chunks]
        
        # 计算句子数（简单估算）
        sentence_counts = []
        for chunk in chunks:
            sentences = chunk.text.count('。') + chunk.text.count('.') + 1
            sentence_counts.append(sentences)
        
        # 计算重叠率（如果有overlap的话）
        overlap_chars = 0
        for i in range(1, len(chunks)):
            # 简单估计：找两个chunk的共同前缀/后缀
            prev_text = chunks[i-1].text[-100:]  # 取前一段的结尾
            curr_text = chunks[i].text[:100]     # 取当前段的开头
            # 计算最长公共子串长度
            overlap = self._calculate_overlap(prev_text, curr_text)
            overlap_chars += overlap
        
        total_chars = sum(chunk_sizes)
        overlap_ratio = overlap_chars / total_chars if total_chars > 0 else 0
        
        return ChunkingMetrics(
            strategy=strategy,
            num_chunks=len(chunks),
            avg_chunk_size=np.mean(chunk_sizes),
            min_chunk_size=min(chunk_sizes),
            max_chunk_size=max(chunk_sizes),
            std_chunk_size=np.std(chunk_sizes),
            total_chars=total_chars,
            processing_time=processing_time,
            avg_sentence_count=np.mean(sentence_counts),
            overlap_ratio=overlap_ratio
        )
    
    def _calculate_overlap(self, text1: str, text2: str) -> int:
        """计算两个文本之间的重叠字符数"""
        max_overlap = 0
        min_len = min(len(text1), len(text2))
        
        for i in range(1, min_len + 1):
            if text1[-i:] == text2[:i]:
                max_overlap = i
        
        return max_overlap
    
    def run_single_experiment(
        self,
        text: str,
        document_name: str,
        strategy: str
    ) -> AblationResult:
        """运行单个实验"""
        print(f"  测试策略: {strategy}...")
        
        # 获取策略参数
        config = self.strategies_config.get(strategy, {})
        
        # 记录处理时间
        start_time = time.time()
        
        try:
            chunks = chunk_document(
                text=text,
                source_id=document_name,
                strategy=strategy,
                **config
            )
        except Exception as e:
            print(f"    错误: {e}")
            chunks = []
        
        processing_time = time.time() - start_time
        
        # 计算指标
        metrics = self.calculate_metrics(chunks, strategy, processing_time)
        
        # 获取样本chunks
        sample_chunks = [c.text[:200] + "..." for c in chunks[:3]]
        
        return AblationResult(
            document_name=document_name,
            document_length=len(text),
            metrics=metrics,
            sample_chunks=sample_chunks
        )
    
    def run_experiment(
        self,
        text: str,
        document_name: str
    ) -> Dict[str, AblationResult]:
        """对单个文档运行所有策略的实验"""
        print(f"\n{'='*70}")
        print(f"文档: {document_name}")
        print(f"长度: {len(text)} 字符")
        print(f"{'='*70}")
        
        results = {}
        
        for strategy in CHUNKING_STRATEGIES.keys():
            result = self.run_single_experiment(text, document_name, strategy)
            results[strategy] = result
        
        return results
    
    def generate_report(
        self,
        all_results: Dict[str, Dict[str, AblationResult]]
    ) -> str:
        """生成实验报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Chunking 策略 Ablation 实验报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 对每个文档生成报告
        for doc_name, results in all_results.items():
            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"文档: {doc_name}")
            report_lines.append(f"{'='*80}")
            
            # 指标对比表
            report_lines.append("\n【指标对比】")
            report_lines.append("-" * 80)
            header = f"{'策略':<20} {'块数':>8} {'平均大小':>10} {'最小':>8} {'最大':>8} {'标准差':>8} {'处理时间':>10}"
            report_lines.append(header)
            report_lines.append("-" * 80)
            
            for strategy, result in results.items():
                m = result.metrics
                line = f"{strategy:<20} {m.num_chunks:>8} {m.avg_chunk_size:>10.1f} {m.min_chunk_size:>8} {m.max_chunk_size:>8} {m.std_chunk_size:>8.1f} {m.processing_time:>10.3f}s"
                report_lines.append(line)
            
            report_lines.append("-" * 80)
            
            # 样本chunks
            report_lines.append("\n【样本Chunks】")
            for strategy, result in results.items():
                report_lines.append(f"\n  策略: {strategy}")
                for i, chunk_sample in enumerate(result.sample_chunks, 1):
                    report_lines.append(f"    Chunk {i}: {chunk_sample}")
        
        # 总体分析
        report_lines.append("\n\n" + "=" * 80)
        report_lines.append("总体分析")
        report_lines.append("=" * 80)
        
        # 计算平均指标
        strategy_avg = {}
        for strategy in CHUNKING_STRATEGIES.keys():
            all_metrics = [r[strategy].metrics for r in all_results.values() if strategy in r]
            if all_metrics:
                strategy_avg[strategy] = {
                    "avg_num_chunks": np.mean([m.num_chunks for m in all_metrics]),
                    "avg_chunk_size": np.mean([m.avg_chunk_size for m in all_metrics]),
                    "avg_processing_time": np.mean([m.processing_time for m in all_metrics]),
                }
        
        report_lines.append("\n【各策略平均表现】")
        report_lines.append("-" * 80)
        report_lines.append(f"{'策略':<20} {'平均块数':>12} {'平均大小':>12} {'平均时间':>12}")
        report_lines.append("-" * 80)
        
        for strategy, avg in strategy_avg.items():
            line = f"{strategy:<20} {avg['avg_num_chunks']:>12.1f} {avg['avg_chunk_size']:>12.1f} {avg['avg_processing_time']:>12.3f}s"
            report_lines.append(line)
        
        report_lines.append("-" * 80)
        
        # 策略特点总结
        report_lines.append("\n【策略特点总结】")
        report_lines.append("""
1. sliding_window (滑动窗口)
   - 优点: 实现简单，速度最快
   - 缺点: 可能截断句子，语义连贯性差

2. token_level (Token级)
   - 优点: 符合LLM的token限制，适合直接输入模型
   - 缺点: 需要tiktoken依赖，可能截断语义

3. semantic (语义分块)
   - 优点: 保持语义连贯，相似句子分组
   - 缺点: 需要sentence-transformers，速度较慢

4. langchain_recursive (LangChain递归)
   - 优点: 优先按段落/句子切分，保持结构完整
   - 缺点: 依赖langchain-text-splitters

5. paragraph_based (段落优先)
   - 优点: 保持段落完整，适合结构化文档
   - 缺点: 块大小不均匀
""")
        
        return "\n".join(report_lines)
    
    def save_results(
        self,
        all_results: Dict[str, Dict[str, AblationResult]],
        report: str
    ):
        """保存实验结果"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存文本报告
        report_path = self.output_dir / f"ablation_report_{timestamp}.txt"
        report_path.write_text(report, encoding='utf-8')
        print(f"\n报告已保存: {report_path}")
        
        # 保存JSON格式结果
        json_results = {}
        for doc_name, results in all_results.items():
            json_results[doc_name] = {
                strategy: {
                    "document_name": r.document_name,
                    "document_length": r.document_length,
                    "metrics": asdict(r.metrics),
                    "sample_chunks": r.sample_chunks
                }
                for strategy, r in results.items()
            }
        
        json_path = self.output_dir / f"ablation_results_{timestamp}.json"
        json_path.write_text(json.dumps(json_results, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"JSON结果已保存: {json_path}")


def run_ablation_experiment(
    pdf_dir: Path = None,
    sample_texts: Dict[str, str] = None
):
    """
    运行Ablation实验的入口函数
    
    Args:
        pdf_dir: PDF文件目录（可选）
        sample_texts: 预设的测试文本（可选）
    """
    experiment = ChunkingAblationExperiment()
    
    all_results = {}
    
    # 使用预设文本或从PDF加载
    if sample_texts:
        test_documents = sample_texts
    elif pdf_dir and pdf_dir.exists():
        # 从PDF加载（简化版，实际应该使用ingest.py）
        print("从PDF加载文档...")
        test_documents = {}
        for pdf_path in pdf_dir.glob("*.pdf"):
            # 这里简化处理，实际应该调用ingest.extract_text_from_pdf
            test_documents[pdf_path.name] = f"[PDF内容: {pdf_path.name}]"
    else:
        # 使用示例文本
        test_documents = {
            "示例文档1": """
第一章 保险条款概述

本保险合同由保险条款、投保单、保险单、保险凭证以及批单等组成。
凡涉及本保险合同的约定，均应采用书面形式。

保险人依照本保险合同的约定，承担赔偿或者给付保险金的责任。
投保人应当按照保险合同约定支付保险费。

本保险合同中的下列术语具有如下含义：
（一）保险人：指与投保人订立保险合同，并按照合同约定承担赔偿或者给付保险金责任的保险公司。
（二）投保人：指与保险人订立保险合同，并按照合同约定负有支付保险费义务的人。
（三）被保险人：指其财产或者人身受保险合同保障，享有保险金请求权的人。
            """ * 5,  # 重复以增大文本量
            "示例文档2": """
能量法求解弹簧的弹性系数

通过力的分解和简化，把弹簧上垂直于簧丝中心线的截面上的应力分布计算出来，
进而采用积分的方法计算出整根弹簧的总应变能，根据能量原理计算得出弹簧的弹性系数。

关键词：截面应力，应变能，弹性势能，弹性系数

引言：弹簧在机械、仪表、电器、交通运输工具以及日常生活器具中广泛应用。
在弹簧的设计计算中，需要考虑的是弹簧的弹性性质与其尺寸结构及材料之间的关系。
            """ * 5,
        }
    
    # 运行实验
    for doc_name, text in test_documents.items():
        results = experiment.run_experiment(text, doc_name)
        all_results[doc_name] = results
    
    # 生成并保存报告
    report = experiment.generate_report(all_results)
    print("\n" + report)
    
    experiment.save_results(all_results, report)
    
    return all_results


if __name__ == "__main__":
    """测试入口"""
    import sys
    
    # 添加项目路径
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 运行实验
    results = run_ablation_experiment()
