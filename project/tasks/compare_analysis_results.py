"""
比较多个模型的分析结果，将不同模型的结果绘制在同一张图上
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from typing import List, Dict, Optional
from pathlib import Path
import logging
import yaml

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_analysis_result(json_path: str) -> Dict:
    """加载分析结果JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_model_name_from_path(json_path: str) -> str:
    """从JSON文件路径中提取模型名称"""
    # 假设路径格式为: results/hidden_states_analysis/model_name/analysis_results.json
    parent_dir = os.path.basename(os.path.dirname(json_path))
    return parent_dir


def load_config(config_path: str) -> Dict:
    """从YAML文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def plot_comparison(
    all_results: Dict[str, Dict],  # {model_name: {question: {keyword: [RR]}}}
    question: str,
    keywords: List[str],
    save_path: Optional[str] = None,
    model_labels: Optional[Dict[str, str]] = None
):
    """
    绘制多个模型的结果比较图
    
    Args:
        all_results: {model_name: {question: {keyword: [RR]}}}
        question: 要比较的问题
        keywords: 要比较的关键词列表
        save_path: 保存路径
        model_labels: 模型名称的显示标签（可选，用于自定义显示名称）
    """
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 确定层数（使用第一个模型的结果）
    first_model = next(iter(all_results.values()))
    if question not in first_model:
        logger.warning(f"Question '{question}' not found in results")
        return
    
    first_keyword = next(iter(first_model[question].keys()))
    num_layers = len(first_model[question][first_keyword])
    layers = [int(i) for i in range(num_layers)]  # 确保是Python int
    
    plt.figure(figsize=(14, 8))
    
    # 为不同模型和关键词分配颜色和标记
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    keyword_markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']
    keyword_linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # 收集所有RR值用于动态调整y轴
    all_rr_values = []
    
    # 为每个模型和关键词绘制
    for model_idx, (model_name, model_results) in enumerate(all_results.items()):
        if question not in model_results:
            continue
        
        # 使用自定义标签或模型名称
        display_name = model_labels.get(model_name, model_name) if model_labels else model_name
        model_color = model_colors[model_idx % len(model_colors)]
        
        for keyword_idx, keyword in enumerate(keywords):
            if keyword not in model_results[question]:
                continue
            
            rrs = model_results[question][keyword]
            # 确保rrs是列表格式
            if isinstance(rrs, list):
                rrs = [float(rr) for rr in rrs]
            else:
                rrs = [float(rr) for rr in list(rrs)]
            all_rr_values.extend(rrs)
            
            marker = keyword_markers[keyword_idx % len(keyword_markers)]
            linestyle = keyword_linestyles[keyword_idx % len(keyword_linestyles)]
            
            # 标签格式：模型名_关键词
            label = f"{display_name}_{keyword}"
            
            # 确保rrs长度与layers匹配，并转换为numpy数组
            if len(rrs) != len(layers):
                min_len = min(len(rrs), len(layers))
                rrs = rrs[:min_len]
                layers_plot = layers[:min_len]
            else:
                layers_plot = layers
            
            # 确保数据是Python原生类型
            rrs_list = [float(rr) for rr in rrs]
            layers_list = [int(l) for l in layers_plot]
            
            plt.plot(
                layers_list, rrs_list,
                marker=marker,
                label=label,
                color=model_color,
                linestyle=linestyle,
                linewidth=2,
                markersize=6,
                markevery=max(1, len(layers_plot) // 20),
                alpha=0.8
            )
    
    plt.xlabel('Layer', fontsize=14, fontweight='bold')
    plt.ylabel('Reciprocal Rank', fontsize=14, fontweight='bold')
    plt.title(f'Reciprocal Rank vs Layer Comparison\nQuestion: {question}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([-0.5, float(num_layers - 0.5)])
    
    # 动态调整y轴范围
    if len(all_rr_values) > 0:
        min_rr = min(all_rr_values)
        max_rr = max(all_rr_values)
        
        # 添加边距
        if max_rr > min_rr:
            margin = (max_rr - min_rr) * 0.1
        elif max_rr > 0:
            margin = max_rr * 0.1
        else:
            margin = 0.01
        
        y_min = float(max(0, min_rr - margin))
        y_max = float(max_rr + margin)
        
        plt.ylim([y_min, y_max])
        
        # 如果值很小，使用科学计数法
        if max_rr < 0.01:
            plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            plt.gca().ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
        else:
            num_ticks = 6
            y_ticks = np.linspace(float(y_min), float(y_max), num_ticks)
            plt.yticks(y_ticks.tolist())
    else:
        plt.ylim(-0.05, 1.05)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # 设置x轴刻度
    if num_layers <= 24:
        plt.xticks(range(0, num_layers))
    else:
        step = max(1, num_layers // 12)
        plt.xticks(range(0, num_layers, step))
    
    plt.tight_layout()
    
    if save_path:
        try:
            # 先绘制图形
            plt.draw()
            # 获取文件格式
            file_format = save_path.split('.')[-1] if '.' in save_path else 'png'
            # 保存
            plt.savefig(save_path, dpi=300, bbox_inches='tight', format=file_format)
            logger.info(f"Comparison plot saved to {save_path}")
        except Exception as e:
            # 如果保存失败，尝试不使用bbox_inches
            try:
                file_format = save_path.split('.')[-1] if '.' in save_path else 'png'
                plt.savefig(save_path, dpi=300, format=file_format)
                logger.info(f"Comparison plot saved to {save_path} (without bbox_inches)")
            except Exception as e2:
                logger.error(f"Failed to save plot: {e2}")
                raise
    
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="比较多个模型的分析结果")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（YAML格式）"
    )
    parser.add_argument(
        "--results",
        type=str,
        nargs='+',
        default=None,
        help="分析结果JSON文件路径列表（可以指定多个）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="要比较的问题（如果不指定，会比较所有问题）"
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs='+',
        default=None,
        help="要比较的关键词列表（如果不指定，会比较所有关键词）"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="模型标签映射文件（JSON格式，{model_name: display_label}）"
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["png", "pdf", "svg"],
        help="输出图片格式"
    )
    
    args = parser.parse_args()
    
    # 从配置文件加载参数（如果提供了配置文件）
    config_dict = {}
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"配置文件不存在: {args.config}")
            return
        logger.info(f"从配置文件加载参数: {args.config}")
        config_dict = load_config(args.config)
    
    # 合并配置：命令行参数优先于配置文件
    results = args.results if args.results is not None else config_dict.get('results', None)
    output = args.output if args.output is not None else config_dict.get('output', 'results/comparison')
    question = args.question if args.question is not None else config_dict.get('question', None)
    keywords = args.keywords if args.keywords is not None else config_dict.get('keywords', None)
    labels = args.labels if args.labels is not None else config_dict.get('labels', None)
    format_type = args.format if args.format is not None else config_dict.get('format', 'png')
    
    # 验证必需参数
    if not results:
        logger.error("必须指定 --results 参数或在配置文件中提供 results 字段")
        parser.print_help()
        return
    
    # 创建输出目录
    os.makedirs(output, exist_ok=True)
    
    # 加载模型标签（如果有）
    model_labels = {}
    if labels:
        if isinstance(labels, dict):
            # 如果labels是字典（从配置文件直接读取）
            model_labels = labels
        else:
            # 如果labels是文件路径
            with open(labels, 'r', encoding='utf-8') as f:
                model_labels = json.load(f)
    
    # 加载所有分析结果
    all_results = {}
    for json_path in results:
        if not os.path.exists(json_path):
            logger.warning(f"File not found: {json_path}, skipping...")
            continue
        
        logger.info(f"Loading results from {json_path}...")
        data = load_analysis_result(json_path)
        model_name = get_model_name_from_path(json_path)
        all_results[model_name] = data['results']
        
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Questions: {list(data['results'].keys())}")
        if model_name in all_results:
            first_question = next(iter(data['results'].keys()))
            logger.info(f"  Keywords: {list(data['results'][first_question].keys())}")
    
    if len(all_results) == 0:
        logger.error("No valid results loaded!")
        return
    
    # 确定要比较的问题和关键词
    first_model = next(iter(all_results.values()))
    questions = [question] if question else list(first_model.keys())
    
    # 确定关键词
    if keywords:
        keywords = keywords if isinstance(keywords, list) else [keywords]
    else:
        # 使用第一个问题的所有关键词
        first_question = next(iter(first_model.keys()))
        keywords = list(first_model[first_question].keys())
    
    logger.info(f"\nComparing {len(all_results)} models:")
    logger.info(f"  Questions: {questions}")
    logger.info(f"  Keywords: {keywords}")
    
    # 为每个问题绘制比较图
    for question in questions:
        # 清理问题文本作为文件名
        safe_question = "".join(c for c in question if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_question = safe_question.replace(' ', '_')[:50]
        
        plot_path = os.path.join(
            output,
            f"comparison_{safe_question}.{format_type}"
        )
        
        try:
            plot_comparison(
                all_results=all_results,
                question=question,
                keywords=keywords,
                save_path=plot_path,
                model_labels=model_labels
            )
        except Exception as e:
            import traceback
            logger.error(f"Failed to plot comparison for question '{question}': {e}")
            logger.error(traceback.format_exc())
    
    logger.info(f"\nComparison completed! Results saved to {output}")


if __name__ == "__main__":
    main()
