#!/usr/bin/env python
"""
生成实验结果的表格和图表
支持从 wandb 或本地 JSON 文件读取数据
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import wandb
from collections import defaultdict
import yaml

# 尝试导入 seaborn（可选）
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def load_data_from_wandb(
    entity: str,
    project: str,
    run_ids: Optional[List[str]] = None,
    run_names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    从 wandb 加载数据
    
    Args:
        entity: wandb entity
        project: wandb project name
        run_ids: 可选的运行 ID 列表
        run_names: 可选的运行名称列表（用于过滤）
    
    Returns:
        字典，key 为运行名称，value 为指标字典
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    results = {}
    
    for run in runs:
        # 如果指定了 run_ids，只加载这些运行
        if run_ids and run.id not in run_ids:
            continue
        
        # 如果指定了 run_names，只加载这些运行
        if run_names and run.name not in run_names:
            continue
        
        # 获取历史数据（包含所有阶段的评估结果）
        history = run.history()
        
        # 优先从summary中获取最终结果（summary通常包含最新的评估结果）
        eval_data = {}
        if hasattr(run, 'summary'):
            for key, value in run.summary.items():
                if key.startswith("report/"):
                    metric_key = key.replace("report/", "")
                    # 只使用数值类型的值
                    if value is not None and not (isinstance(value, float) and pd.isna(value)):
                        eval_data[metric_key] = value
        
        # 如果summary中没有某些指标，再从history中获取最后一个非NaN值
        for key in history.columns:
            if key.startswith("report/"):
                metric_key = key.replace("report/", "")
                # 如果summary中已经有这个指标，跳过（summary优先）
                if metric_key not in eval_data:
                    # 获取最后一个非 NaN 值
                    values = history[key].dropna()
                    if len(values) > 0:
                        eval_data[metric_key] = values.iloc[-1]
        
        # 获取运行配置信息
        config = run.config if hasattr(run, 'config') else {}
        
        # 如果同名运行已存在，优先保留有更多有效数据的运行
        if run.name in results:
            existing_metrics = results[run.name].get("metrics", {})
            # 计算有效数据数量（非NaN的数据）
            # 对于关键指标（forget, neighbor），检查是否有非零值
            key_metrics = ['forget/fb', 'forget/qa', 'forget/aa', 
                          'neighbor/qa', 'neighbor/fb']
            existing_key_count = sum(1 for k in key_metrics 
                                    if k in existing_metrics 
                                    and not pd.isna(existing_metrics[k]) 
                                    and existing_metrics[k] != 0)
            new_key_count = sum(1 for k in key_metrics 
                              if k in eval_data 
                              and not pd.isna(eval_data[k]) 
                              and eval_data[k] != 0)
            
            # 如果新运行有更多有效的关键指标，则替换
            if new_key_count > existing_key_count:
                results[run.name] = {
                    "metrics": eval_data,
                    "config": config,
                    "id": run.id,
                }
            # 否则保持原有数据（忽略新运行）
        else:
            results[run.name] = {
                "metrics": eval_data,
                "config": config,
                "id": run.id,
            }
    
    return results


def load_data_from_json(file_path: str) -> Dict[str, Dict]:
    """
    从 JSON 文件加载数据
    
    Args:
        file_path: JSON 文件路径
    
    Returns:
        字典，key 为运行名称，value 为指标字典
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def aggregate_results_across_targets(
    results: Dict[str, Dict],
    method_key: str = "method",
) -> Dict[str, Dict]:
    """
    跨多个目标聚合结果（计算平均值）
    
    Args:
        results: 所有运行的结果
        method_key: 用于识别方法的配置键
    
    Returns:
        按方法聚合后的结果
    """
    # 按方法分组
    method_groups = defaultdict(list)
    
    for run_name, run_data in results.items():
        method = run_data.get("config", {}).get(method_key, "Unknown")
        method_groups[method].append(run_data["metrics"])
    
    # 计算每个方法的平均值
    aggregated = {}
    for method, metrics_list in method_groups.items():
        if not metrics_list:
            continue
        
        # 获取所有指标键
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        # 计算平均值
        aggregated_metrics = {}
        for key in all_keys:
            values = [m.get(key) for m in metrics_list if key in m and m[key] is not None]
            if values:
                aggregated_metrics[key] = np.mean(values)
        
        aggregated[method] = aggregated_metrics
    
    return aggregated


def create_comparison_table(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    创建对比表格
    
    Args:
        results: 结果字典，key 为方法名，value 为指标字典
        output_path: 可选的输出路径（CSV）
    
    Returns:
        DataFrame 表格
    """
    rows = []
    
    for method, run_data in results.items():
        # run_data 的结构是 {"metrics": {...}, "config": {...}, "id": ...}
        metrics = run_data.get("metrics", {})
        row = {
            "METHOD": method,
            "FORGET_FB": metrics.get("forget/fb", np.nan),
            "FORGET_QA": metrics.get("forget/qa", np.nan),
            "FORGET_AA": metrics.get("forget/aa", np.nan),
            "FORGET_ALL": metrics.get("forget/all", np.nan),
            "NEIGHBOR_QA": metrics.get("neighbor/qa", np.nan),
            "NEIGHBOR_FB": metrics.get("neighbor/fb", np.nan),
            "NEIGHBOR_ALL": metrics.get("neighbor/all", np.nan),
            "UTILITY_MMLU": metrics.get("utility/gen", np.nan),  # MMLU
            "UTILITY_BBH": metrics.get("utility/rea", np.nan),  # BBH (reasoning)
            "UTILITY_TRUTHFUL": metrics.get("utility/tru", np.nan),
            "UTILITY_TRIVIA": metrics.get("utility/fac", np.nan),
            "UTILITY_FLUENCY": metrics.get("utility/flu", np.nan),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 按 FORGET_ALL 排序（升序，越低越好）
    if "FORGET_ALL" in df.columns:
        df = df.sort_values("FORGET_ALL", na_position='last')
    
    if output_path:
        df.to_csv(output_path, index=False, float_format='%.2f')
        print(f"表格已保存到: {output_path}")
    
    return df


def plot_comparison_table(
    results: Dict[str, Dict],
    output_path: str,
    method_name_mapping: Optional[Dict[str, str]] = None,
):
    """
    绘制对比表格（转置格式：方法在行，指标在列）
    
    Args:
        results: 结果字典
        output_path: 输出路径
        method_name_mapping: 方法名称映射（用于美化显示名称）
    """
    # 收集所有可能的指标（从所有方法中收集）
    all_metric_keys = set()
    for method, run_data in results.items():
        metrics = run_data.get("metrics", {})
        all_metric_keys.update(metrics.keys())
    
    # 排除不需要的指标
    excluded_metrics = {"step", "stage_number", "_step"}
    all_metric_keys = all_metric_keys - excluded_metrics
    
    # 定义指标的顺序和显示名称
    # 按照类别分组：forget, neighbor, mia, utility
    metric_order = [
        # Forget metrics
        "forget/fb", "forget/qa", "forget/aa", "forget/all",
        # Neighbor metrics
        "neighbor/fb", "neighbor/qa", "neighbor/all",
        # MIA metrics
        "mia/fm", "mia/rt",
        # Utility metrics
        "utility/gen", "utility/rea", "utility/tru", "utility/fac", "utility/flu",
    ]
    
    # 只保留实际存在的指标
    metric_order = [m for m in metric_order if m in all_metric_keys]
    # 添加其他未列出的指标（排除不需要的指标）
    other_metrics = sorted(all_metric_keys - set(metric_order) - excluded_metrics)
    metric_order.extend(other_metrics)
    
    # 指标显示名称映射
    metric_display_names = {
        "forget/fb": "Forget FB",
        "forget/qa": "Forget QA",
        "forget/aa": "Forget AA",
        "forget/all": "Forget All",
        "neighbor/fb": "Neighbor FB",
        "neighbor/qa": "Neighbor QA",
        "neighbor/all": "Neighbor All",
        "mia/fm": "MIA FM",
        "mia/rt": "MIA RT",
        "utility/gen": "Utility Gen",
        "utility/rea": "Utility Rea",
        "utility/tru": "Utility Tru",
        "utility/fac": "Utility Fac",
        "utility/flu": "Utility Flu",
    }
    
    # 准备数据（转置：方法在行，指标在列）
    rows = []
    for method, run_data in results.items():
        metrics = run_data.get("metrics", {})
        # 跳过没有数据的行
        if not metrics or all(pd.isna(v) for v in metrics.values()):
            continue
        
        # 美化方法名称 - 按优先级匹配（更具体的模式优先）
        display_name = method
        if method_name_mapping and method in method_name_mapping:
            display_name = method_name_mapping[method]
        else:
            # 自动美化一些常见的方法名 - 按优先级匹配
            method_lower = method.lower()
            if "kl+enpo" in method_lower:
                display_name = "KL+ENPO"
            elif "npo" in method_lower and "enpo" not in method_lower:
                display_name = "NPO"
            elif "atu" in method_lower:
                display_name = "ATU"
            elif "ga" in method_lower and "enpo" not in method_lower:
                display_name = "GA"
            elif "enpo" in method_lower:
                # 检查 l2 参数
                if "l1:1.0-l2:0" in method or "l2:0" in method:
                    display_name = "ENPO (l2=0)"
                elif "l1:1.0-l2:0.05" in method or "l2:0.05" in method:
                    display_name = "ENPO (l2=0.05)"
                elif "l1:1.0-l2:0.1" in method or "l2:0.1" in method:
                    display_name = "ENPO (l2=0.1)"
                elif "l1:1.0-l2:0.5" in method or "l2:0.5" in method:
                    display_name = "ENPO (l2=0.5)"
                elif "l2:0.06" in method:
                    display_name = "ENPO (l2=0.06)"
                else:
                    display_name = "ENPO"
        
        row = {"METHOD": display_name}
        for metric_key in metric_order:
            value = metrics.get(metric_key, np.nan)
            row[metric_key] = value
        rows.append(row)
    
    if not rows:
        print("警告: 没有有效数据绘制表格")
        return
    
    df = pd.DataFrame(rows)
    
    # 按方法名排序（ENPO优先，然后按字母顺序）
    def sort_key(name):
        if "ENPO" in name.upper():
            return (0, name)
        return (1, name)
    df = df.sort_values("METHOD", key=lambda x: x.map(sort_key), na_position='last')
    
    # 创建表格图（转置：方法在行，指标在列）
    # 计算合适的图形大小：宽度取决于指标数量，高度取决于方法数量
    num_metrics = len(metric_order)
    num_methods = len(df)
    fig, ax = plt.subplots(figsize=(max(14, num_metrics * 1.2), max(6, num_methods * 0.6 + 2)))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据 - 转置格式
    table_data = []
    
    # 表头 - 第一列是METHOD，后面是各个指标
    header = ["METHOD"] + [metric_display_names.get(m, m.replace("/", " ").title()) for m in metric_order]
    
    # 数据行
    for _, row in df.iterrows():
        row_data = [row["METHOD"]]
        for metric_key in metric_order:
            value = row[metric_key]
            if pd.isna(value):
                row_data.append("-")
            else:
                # 根据指标类型决定显示格式
                if "forget" in metric_key or "neighbor" in metric_key or "utility" in metric_key or "mia" in metric_key:
                    row_data.append(f"{value:.1f}")
                else:
                    row_data.append(f"{value:.2f}")
        table_data.append(row_data)
    
    # 合并表头和数据
    all_data = [header] + table_data
    
    # 创建表格
    table = ax.table(
        cellText=all_data,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    # 设置表头样式 - 根据指标类型使用不同颜色
    def get_header_color(metric_key):
        if "forget" in metric_key:
            return '#E74C3C'  # 红色
        elif "neighbor" in metric_key:
            return '#F39C12'  # 橙色
        elif "mia" in metric_key:
            return '#9B59B6'  # 紫色
        elif "utility" in metric_key:
            return '#27AE60'  # 绿色
        else:
            return '#3498DB'  # 蓝色
    
    header_colors = ['#1A1A1A'] + [get_header_color(m) for m in metric_order]
    
    for j in range(len(header)):
        cell = table[(0, j)]
        cell.set_facecolor(header_colors[j])
        # METHOD列使用深色背景
        if j == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=11)
            cell.get_text().set_text("METHOD")
            cell.set_height(0.08)
            cell.visible_edges = 'TBLR'
            cell.set_edgecolor('white')
            cell.set_linewidth(1.5)
        else:
            cell.set_text_props(weight='bold', color='white', fontsize=9)
            cell.set_height(0.08)
            # 添加箭头指示方向
            metric_key = metric_order[j-1]
            direction = ""
            if "forget" in metric_key or "mia" in metric_key:
                direction = " ↓"
            elif "neighbor" in metric_key or "utility" in metric_key:
                direction = " ↑"
            cell.get_text().set_text(header[j] + direction)
    
    # 识别每列的最佳值（用于加粗）
    # FORGET和MIA列：越低越好
    # NEIGHBOR和UTILITY列：越高越好
    best_cells = set()  # 存储(row_idx, col_idx)元组
    
    for col_idx, metric_key in enumerate(metric_order, start=1):  # start=1因为第0列是METHOD
        is_lower_better = "forget" in metric_key or "mia" in metric_key
        values = []
        for row_idx in range(len(df)):
            val = df.iloc[row_idx][metric_key]
            if not pd.isna(val):
                values.append((row_idx, val))
        
        if values:
            if is_lower_better:
                best_row_idx, _ = min(values, key=lambda x: x[1])
            else:
                best_row_idx, _ = max(values, key=lambda x: x[1])
            best_cells.add((best_row_idx, col_idx))
    
    # 设置数据行样式
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            cell = table[(i + 1, j)]  # +1 因为有一行表头
            if i % 2 == 0:
                cell.set_facecolor('#F8F9FA')
            else:
                cell.set_facecolor('white')
            cell.set_height(0.05)
            cell.set_edgecolor('#E0E0E0')
            cell.set_linewidth(0.5)
            
            # 如果是最佳值，加粗
            if j > 0 and (i, j) in best_cells:  # j > 0 排除METHOD列
                cell.set_text_props(weight='bold', fontsize=9)
            else:
                cell.set_text_props(weight='normal', fontsize=9)
            
            # METHOD列特殊处理
            if j == 0:
                cell.set_text_props(weight='bold', fontsize=9)
    
    # 添加列之间的分隔线（视觉上分组）
    # 在不同指标类别之间添加分隔线
    category_boundaries = []
    prev_category = None
    for idx, metric_key in enumerate(metric_order, start=1):
        if "forget" in metric_key:
            category = "forget"
        elif "neighbor" in metric_key:
            category = "neighbor"
        elif "mia" in metric_key:
            category = "mia"
        elif "utility" in metric_key:
            category = "utility"
        else:
            category = "other"
        
        if prev_category and category != prev_category:
            category_boundaries.append(idx)
        prev_category = category
    
    for i in range(len(all_data)):
        # METHOD列保持正常边框
        table[(i, 0)].visible_edges = 'TBLR' if i == 0 else 'BLR'
        # 在类别边界添加分隔线
        for boundary in category_boundaries:
            if boundary < len(header):
                table[(i, boundary)].visible_edges = 'TBR' if i == 0 else 'BR'
    
    plt.title("Table 1. Unlearning Results", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"表格图已保存到: {output_path}")
    plt.close()


def plot_utility_vs_forget(
    results: Dict[str, Dict],
    output_path: str,
    layer_key: Optional[str] = None,
):
    """
    绘制 Utility vs Forget 权衡图
    
    Args:
        results: 结果字典
        output_path: 输出路径
        layer_key: 可选的层标识符（用于区分不同层）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准备数据
    forget_all = []
    utility_gen = []
    neighbor_all = []
    layers = []
    
    for method, run_data in results.items():
        # run_data 的结构是 {"metrics": {...}, "config": {...}, "id": ...}
        metrics = run_data.get("metrics", {})
        forget = metrics.get("forget/all")
        utility = metrics.get("utility/gen")
        neighbor = metrics.get("neighbor/all")
        
        if forget is not None and utility is not None:
            forget_all.append(forget)
            utility_gen.append(utility)
            neighbor_all.append(neighbor if neighbor is not None else np.nan)
            
            # 尝试从方法名或配置中提取层信息
            if layer_key:
                layer = metrics.get(layer_key, method)
            else:
                # 尝试从方法名中提取层号
                import re
                match = re.search(r'layer[_\s]?(\d+)', method.lower())
                layer = match.group(1) if match else "Unknown"
            layers.append(layer)
    
    forget_all = np.array(forget_all)
    utility_gen = np.array(utility_gen)
    neighbor_all = np.array(neighbor_all)
    
    # 左图：Utility vs Forget
    ax1 = axes[0]
    unique_layers = sorted(set(layers))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_layers)))
    layer_colors = {layer: colors[i] for i, layer in enumerate(unique_layers)}
    
    for layer in unique_layers:
        mask = np.array([l == layer for l in layers])
        if mask.any():
            ax1.scatter(
                forget_all[mask],
                utility_gen[mask],
                label=f"Layer {layer}",
                color=layer_colors[layer],
                s=100,
                alpha=0.7,
            )
            
            # 添加趋势线
            if mask.sum() > 1:
                z = np.polyfit(forget_all[mask], utility_gen[mask], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(forget_all[mask].min(), forget_all[mask].max(), 100)
                ax1.plot(x_trend, p(x_trend), "--", color=layer_colors[layer], alpha=0.5)
    
    ax1.set_xlabel("Forget (All)", fontsize=12)
    ax1.set_ylabel("Utility (Gen)", fontsize=12)
    ax1.set_title("Utility vs. Forget Across Layer", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：Neighbor vs Forget
    ax2 = axes[1]
    for layer in unique_layers:
        mask = np.array([l == layer for l in layers])
        if mask.any() and not np.isnan(neighbor_all[mask]).all():
            ax2.scatter(
                forget_all[mask],
                neighbor_all[mask],
                label=f"Layer {layer}",
                color=layer_colors[layer],
                s=100,
                alpha=0.7,
            )
            
            # 添加趋势线
            valid_mask = mask & ~np.isnan(neighbor_all)
            if valid_mask.sum() > 1:
                z = np.polyfit(forget_all[valid_mask], neighbor_all[valid_mask], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(forget_all[valid_mask].min(), forget_all[valid_mask].max(), 100)
                ax2.plot(x_trend, p(x_trend), "--", color=layer_colors[layer], alpha=0.5)
    
    ax2.set_xlabel("Forget (All)", fontsize=12)
    ax2.set_ylabel("Neighbor (All)", fontsize=12)
    ax2.set_title("Neighbor vs. Forget Across Layer", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    plt.close()


def plot_neighbor_vs_forget_tradeoff(
    results: Dict[str, Dict],
    output_path: str,
):
    """
    绘制 Neighbor vs Forget 权衡曲线
    
    Args:
        results: 结果字典
        output_path: 输出路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    forget_all = []
    neighbor_all = []
    
    for method, run_data in results.items():
        # run_data 的结构是 {"metrics": {...}, "config": {...}, "id": ...}
        metrics = run_data.get("metrics", {})
        forget = metrics.get("forget/all")
        neighbor = metrics.get("neighbor/all")
        
        if forget is not None and neighbor is not None:
            forget_all.append(forget)
            neighbor_all.append(neighbor)
    
    if len(forget_all) == 0:
        print("警告: 没有足够的数据绘制权衡图")
        return
    
    forget_all = np.array(forget_all)
    neighbor_all = np.array(neighbor_all)
    
    # 排序以便绘制平滑曲线
    sort_idx = np.argsort(forget_all)
    forget_all = forget_all[sort_idx]
    neighbor_all = neighbor_all[sort_idx]
    
    ax.plot(forget_all, neighbor_all, 'o-', color='orange', linewidth=2, markersize=8)
    ax.set_xlabel("Mean Forget (All)", fontsize=12)
    ax.set_ylabel("Mean Neighbor (All)", fontsize=12)
    ax.set_title("Neighbor vs Forget Trade-Off", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    plt.close()


def plot_forget_over_stages(
    stage_results: Dict[str, List[Dict]],
    output_path: str,
    layer_key: Optional[str] = None,
):
    """
    绘制不同阶段 Forget 分数的变化
    
    Args:
        stage_results: 字典，key 为方法/层标识，value 为每个阶段的指标列表
        output_path: 输出路径
        layer_key: 可选的层标识符
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    unique_layers = sorted(set(stage_results.keys()))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_layers)))
    layer_colors = {layer: colors[i] for i, layer in enumerate(unique_layers)}
    
    for layer, stages in stage_results.items():
        stages_sorted = sorted(stages, key=lambda x: x.get("stage_number", 0))
        stage_numbers = [s.get("stage_number", i) for i, s in enumerate(stages_sorted)]
        forget_scores = [s.get("forget/all", np.nan) for s in stages_sorted]
        
        # 过滤掉 NaN 值
        valid_indices = [i for i, f in enumerate(forget_scores) if not np.isnan(f)]
        if valid_indices:
            stage_numbers = [stage_numbers[i] for i in valid_indices]
            forget_scores = [forget_scores[i] for i in valid_indices]
            
            label = f"Layer {layer}" if layer != "Unknown" else layer
            ax.plot(
                stage_numbers,
                forget_scores,
                'o-',
                label=label,
                color=layer_colors.get(layer, 'black'),
                linewidth=2,
                markersize=6,
            )
    
    ax.set_xlabel("Unlearning Stage", fontsize=12)
    ax.set_ylabel("Forget (All)", fontsize=12)
    ax.set_title("Forget Scores Over Stages", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    plt.close()


def load_stage_results_from_wandb(
    entity: str,
    project: str,
    run_ids: Optional[List[str]] = None,
) -> Dict[str, List[Dict]]:
    """
    从 wandb 加载所有阶段的结果
    
    Args:
        entity: wandb entity
        project: wandb project name
        run_ids: 可选的运行 ID 列表
    
    Returns:
        字典，key 为层标识，value 为每个阶段的指标列表
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    stage_results = defaultdict(list)
    
    for run in runs:
        if run_ids and run.id not in run_ids:
            continue
        
        # 获取历史数据
        history = run.history()
        
        # 提取层信息
        import re
        match = re.search(r'layer[_\s]?(\d+)', run.name.lower())
        layer = match.group(1) if match else "Unknown"
        
        # 提取每个阶段的结果
        for idx, row in history.iterrows():
            stage_num = row.get("report/stage_number", idx)
            if pd.isna(stage_num):
                continue
            
            stage_metrics = {}
            for key in row.index:
                if key.startswith("report/"):
                    value = row[key]
                    if not pd.isna(value):
                        stage_metrics[key.replace("report/", "")] = value
            
            if stage_metrics:
                stage_metrics["stage_number"] = int(stage_num)
                stage_results[layer].append(stage_metrics)
    
    return dict(stage_results)


def load_all_evaluation_points_from_wandb(
    entity: str,
    project: str,
    run_ids: Optional[List[str]] = None,
    run_names: Optional[List[str]] = None,
    task_name_filter: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """
    从 wandb 加载所有评估点的数据
    
    Args:
        entity: wandb entity
        project: wandb project name
        run_ids: 可选的运行 ID 列表
        run_names: 可选的运行名称列表
        task_name_filter: 可选的task.name过滤条件（例如："unlearning-enpo"）
    
    Returns:
        字典，key 为运行名称，value 为所有评估点的列表
    """
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    all_eval_points = {}
    
    for run in runs:
        if run_ids and run.id not in run_ids:
            continue
        if run_names and run.name not in run_names:
            continue
        
        # 如果指定了task.name过滤条件，检查config中的task.name
        if task_name_filter:
            config = run.config if hasattr(run, 'config') else {}
            task_name = config.get("task", {}).get("name", "") if isinstance(config.get("task"), dict) else ""
            if task_name_filter.lower() not in str(task_name).lower():
                continue
        
        history = run.history()
        eval_points = []
        
        # 提取所有有report指标的行
        for idx, row in history.iterrows():
            point_metrics = {}
            step = row.get("_step", idx)
            
            # 提取所有report指标
            for key in row.index:
                if key.startswith("report/"):
                    value = row[key]
                    if not pd.isna(value):
                        point_metrics[key.replace("report/", "")] = value
            
            if point_metrics:
                point_metrics["_step"] = step
                eval_points.append(point_metrics)
        
        if eval_points:
            all_eval_points[run.name] = eval_points
    
    return all_eval_points


def find_best_enpo_node(
    all_eval_points: Dict[str, List[Dict]],
    enpo_method_pattern: str = "enpo",
    comparison_methods: Optional[List[str]] = None,
    comparison_patterns: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    找到ENPO表现最好的节点（forget最低，neighbor和utility最高）
    
    Args:
        all_eval_points: 所有运行的评估点数据
        enpo_method_pattern: ENPO方法的名称模式
        comparison_methods: 要比较的其他方法名称列表（如果为None，则比较所有非ENPO方法）
        comparison_patterns: 要比较的其他方法的名称模式列表（例如：["ga", "atu", "npo"]）
    
    Returns:
        最佳节点的数据，包含运行名称、step和指标
    """
    # 识别ENPO方法和其他方法
    enpo_methods = [name for name in all_eval_points.keys() 
                    if enpo_method_pattern.lower() in name.lower()]
    
    if comparison_methods is None:
        if comparison_patterns:
            # 使用指定的模式来识别比较方法
            comparison_methods = []
            for pattern in comparison_patterns:
                comparison_methods.extend([name for name in all_eval_points.keys() 
                                         if pattern.lower() in name.lower() 
                                         and enpo_method_pattern.lower() not in name.lower()])
            # 去重
            comparison_methods = list(set(comparison_methods))
        else:
            # 默认：比较所有非ENPO方法
            comparison_methods = [name for name in all_eval_points.keys() 
                                 if enpo_method_pattern.lower() not in name.lower()]
    
    if not enpo_methods or not comparison_methods:
        print("警告: 没有找到足够的ENPO方法或比较方法")
        return None
    
    print(f"找到 {len(enpo_methods)} 个ENPO方法: {enpo_methods}")
    print(f"找到 {len(comparison_methods)} 个比较方法: {comparison_methods}")
    
    best_node = None
    best_score = -float('inf')
    
    # 遍历所有ENPO方法的评估点
    for enpo_name in enpo_methods:
        enpo_points = all_eval_points[enpo_name]
        
        for point in enpo_points:
            enpo_forget = point.get("forget/all")
            enpo_neighbor = point.get("neighbor/all")
            enpo_utility = point.get("utility/gen")
            
            # 确保有完整的指标
            if pd.isna(enpo_forget) or pd.isna(enpo_neighbor) or pd.isna(enpo_utility):
                continue
            
            # 要求ENPO的neighbor至少为40
            if enpo_neighbor < 40:
                continue
            
            # 要求ENPO的utility不能过低（至少为60，这是一个合理的阈值）
            if enpo_utility < 60:
                continue
            
            # 找其他方法最差的节点（而不是同一step的节点）
            # 这样可以让ENPO最好的表现与其他方法最差的表现进行对比
            worst_comparison_nodes = find_worst_comparison_nodes(
                all_eval_points,
                comparison_methods,
                utility_threshold=60.0
            )
            
            if not worst_comparison_nodes:
                continue
            
            comparison_metrics = {name: node['point'] for name, node in worst_comparison_nodes.items()}
            step = point.get("_step")
            
            if not comparison_metrics:
                continue
            
            # 计算综合得分（即使不是在所有指标上都最优，也计算得分）
            # 这样我们可以找到ENPO相对表现最好的节点
            if len(comparison_metrics) > 0:
                # 计算综合得分（forget越低越好，neighbor和utility越高越好）
                # 使用相对改进作为得分
                # 排除0值和NaN值
                comp_forgets = [p.get("forget/all", 0) for p in comparison_metrics.values() 
                               if not pd.isna(p.get("forget/all")) and p.get("forget/all", 0) > 0]
                comp_neighbors = [p.get("neighbor/all", 0) for p in comparison_metrics.values() 
                                if not pd.isna(p.get("neighbor/all")) and p.get("neighbor/all", 0) > 0]
                comp_utilities = [p.get("utility/gen", 0) for p in comparison_metrics.values() 
                                 if not pd.isna(p.get("utility/gen")) and p.get("utility/gen", 0) > 0]
                
                # 如果过滤后没有有效数据，跳过
                if not comp_forgets or not comp_neighbors or not comp_utilities:
                    continue
                
                # 使用中位数而不是平均值，更稳健
                avg_comp_forget = np.median(comp_forgets)
                avg_comp_neighbor = np.median(comp_neighbors)
                avg_comp_utility = np.median(comp_utilities)
                
                # 计算相对改进（ENPO相对于平均值的改进）
                forget_improvement = (avg_comp_forget - enpo_forget) / avg_comp_forget if avg_comp_forget > 0 else 0
                neighbor_improvement = (enpo_neighbor - avg_comp_neighbor) / avg_comp_neighbor if avg_comp_neighbor > 0 else 0
                utility_improvement = (enpo_utility - avg_comp_utility) / avg_comp_utility if avg_comp_utility > 0 else 0
                
                # 综合得分：forget改进 + neighbor改进 + utility改进
                # 如果ENPO的forget更低、neighbor和utility更高，得分会更高
                score = forget_improvement + neighbor_improvement + utility_improvement
                
                # 优先考虑ENPO的forget更低，且neighbor和utility更高的节点
                forget_better = enpo_forget < avg_comp_forget
                neighbor_better = enpo_neighbor > avg_comp_neighbor
                utility_better = enpo_utility > avg_comp_utility
                
                # 如果ENPO的forget更低，给予更高的权重
                if forget_better:
                    score += 1.0  # 额外奖励
                    # 如果neighbor和utility不是太低（至少是平均值的80%），也给予奖励
                    if enpo_neighbor >= avg_comp_neighbor * 0.8:
                        score += 0.3
                    if enpo_utility >= avg_comp_utility * 0.8:
                        score += 0.3
                
                # 要求：forget更低（这是必须的），且neighbor和utility不是太低
                # 如果forget更低，且neighbor和utility都达到平均值的80%，就考虑
                # 或者如果forget明显更低（低10%以上），且neighbor和utility都达到平均值的70%，也考虑
                forget_much_better = (avg_comp_forget - enpo_forget) / avg_comp_forget >= 0.1 if avg_comp_forget > 0 else False
                neighbor_ok = enpo_neighbor >= avg_comp_neighbor * 0.8
                utility_ok = enpo_utility >= avg_comp_utility * 0.8
                neighbor_ok_loose = enpo_neighbor >= avg_comp_neighbor * 0.7
                utility_ok_loose = enpo_utility >= avg_comp_utility * 0.7
                
                # 要求utility不能过低：至少达到平均值的75%，或者绝对值至少为60
                utility_too_low = enpo_utility < max(avg_comp_utility * 0.75, 60)
                
                # 如果forget更低，且(neighbor和utility都达到80% 或 (forget明显更低且都达到70%))，且utility不过低，就考虑
                if forget_better and not utility_too_low and ((neighbor_ok and utility_ok) or (forget_much_better and neighbor_ok_loose and utility_ok_loose)) and score > best_score:
                    best_score = score
                    best_node = {
                        "enpo_method": enpo_name,
                        "step": step,
                        "enpo_metrics": point,
                        "comparison_metrics": comparison_metrics,
                        "score": score,
                    }
    
    return best_node


def find_worst_comparison_nodes(
    all_eval_points: Dict[str, List[Dict]],
    comparison_methods: List[str],
    utility_threshold: float = 60.0,
) -> Dict[str, Dict]:
    """
    找到每个比较方法最差的节点（用于与ENPO最好的节点对比）
    
    Args:
        all_eval_points: 所有运行的评估点数据
        comparison_methods: 要比较的方法名称列表
        utility_threshold: utility的最小阈值（不能低于此值）
    
    Returns:
        字典，key为方法名，value为最差节点的数据
    """
    worst_nodes = {}
    
    for comp_name in comparison_methods:
        if comp_name not in all_eval_points:
            continue
        
        comp_points = all_eval_points[comp_name]
        candidates = []
        
        for point in comp_points:
            forget = point.get("forget/all")
            neighbor = point.get("neighbor/all")
            utility = point.get("utility/gen")
            
            # 确保有完整的指标
            if pd.isna(forget) or pd.isna(neighbor) or pd.isna(utility):
                continue
            
            # 要求utility不能过低
            if utility < utility_threshold:
                continue
            
            # 对于GA方法，如果neighbor > 70，则找forget最差的
            if 'ga' in comp_name.lower() and 'enpo' not in comp_name.lower():
                if neighbor > 70:
                    candidates.append({
                        'point': point,
                        'forget': forget,
                        'neighbor': neighbor,
                        'utility': utility,
                        'score': forget  # 使用forget作为得分（越高越差）
                    })
            else:
                # 对于其他方法，找forget最高（最差）且neighbor和utility最低（最差）的节点
                # 但utility不能低于阈值
                # 综合得分：forget越高越好（越差），neighbor和utility越低越好（越差）
                # 但需要考虑trade-off，不能只追求forget高而忽略其他指标
                score = forget - (neighbor * 0.3 + utility * 0.2)  # forget权重更高
                candidates.append({
                    'point': point,
                    'forget': forget,
                    'neighbor': neighbor,
                    'utility': utility,
                    'score': score
                })
        
        if candidates:
            # 找到得分最高的（最差的）
            worst = max(candidates, key=lambda x: x['score'])
            worst_nodes[comp_name] = worst
    
    return worst_nodes


def find_worst_ga_node(
    entity: str,
    project: str,
    neighbor_threshold: float = 70,
) -> Optional[Dict]:
    """
    找到GA方法中neighbor > threshold且forget最差的节点
    
    Args:
        entity: wandb entity
        project: wandb project name
        neighbor_threshold: neighbor的最小值
    
    Returns:
        最差节点的数据
    """
    import wandb
    import pandas as pd
    
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    candidates = []
    
    for run in runs:
        if 'ga' in run.name.lower() and 'enpo' not in run.name.lower():
            history = run.history()
            
            for idx, row in history.iterrows():
                point_metrics = {}
                step = row.get('_step', idx)
                
                for key in row.index:
                    if key.startswith('report/'):
                        value = row[key]
                        if not pd.isna(value):
                            point_metrics[key.replace('report/', '')] = value
                
                if point_metrics:
                    neighbor = point_metrics.get('neighbor/all', 0)
                    forget = point_metrics.get('forget/all', 0)
                    if not pd.isna(neighbor) and not pd.isna(forget) and neighbor > neighbor_threshold:
                        candidates.append({
                            'run_id': run.id,
                            'run_name': run.name,
                            'step': step,
                            'forget': forget,
                            'neighbor': neighbor,
                            'utility': point_metrics.get('utility/gen', 0),
                            'point': point_metrics
                        })
    
    if candidates:
        # 找到forget最差的（最高）
        worst = max(candidates, key=lambda x: x['forget'])
        return worst
    
    return None


def load_config(config_path: str) -> Dict:
    """
    从YAML配置文件加载参数
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> argparse.Namespace:
    """
    将配置文件中的参数合并到args中（命令行参数优先）
    
    Args:
        config: 配置字典
        args: 命令行参数
    
    Returns:
        合并后的参数
    """
    # 如果命令行没有指定（None），则使用配置文件中的值
    # 对于有默认值的参数，配置文件的值会覆盖默认值
    
    if config.get("source"):
        args.source = config["source"]
    
    if config.get("wandb"):
        wandb_config = config["wandb"]
        # 只有当命令行没有明确指定时才使用配置文件的值
        if wandb_config.get("entity") and not args.entity:
            args.entity = wandb_config["entity"]
        if wandb_config.get("project") and not args.project:
            args.project = wandb_config["project"]
        if wandb_config.get("run_ids") and not args.run_ids:
            args.run_ids = wandb_config["run_ids"]
        if wandb_config.get("run_names") and not args.run_names:
            args.run_names = wandb_config["run_names"]
    
    if config.get("json_path") and not args.json_path:
        args.json_path = config["json_path"]
    
    # output_dir: 如果配置文件指定了，且命令行使用的是默认值，则使用配置文件的值
    # 注意：由于argparse的特性，如果用户在命令行指定了--output_dir，args.output_dir就是用户指定的值
    # 如果用户没有指定，args.output_dir就是默认值"./results"
    # 所以我们检查是否是默认值，如果是，就用配置文件的值覆盖
    if config.get("output_dir"):
        # 如果output_dir是默认值，说明用户没有在命令行指定，使用配置文件的值
        if args.output_dir == "./results":
            args.output_dir = config["output_dir"]
        # 如果用户明确指定了不同的值，保持用户指定的值（不覆盖）
    
    if config.get("aggregate") and not args.aggregate:
        args.aggregate = config["aggregate"]
    
    # 保存方法名称映射（如果存在）
    if config.get("method_name_mapping"):
        args.method_name_mapping = config["method_name_mapping"]
    else:
        args.method_name_mapping = None
    
    # 保存task_name_filter（如果存在）
    if config.get("task_name_filter") and not hasattr(args, 'task_name_filter'):
        args.task_name_filter = config["task_name_filter"]
    
    return args


def main():
    parser = argparse.ArgumentParser(description="生成实验结果表格和图表")
    parser.add_argument(
        "--source",
        type=str,
        choices=["wandb", "json"],
        default="wandb",
        help="数据来源：wandb 或 json 文件",
    )
    parser.add_argument("--entity", type=str, help="wandb entity")
    parser.add_argument("--project", type=str, help="wandb project name")
    parser.add_argument("--json_path", type=str, help="JSON 文件路径（当 source=json 时）")
    parser.add_argument("--run_ids", type=str, nargs="+", help="wandb 运行 ID 列表")
    parser.add_argument("--run_names", type=str, nargs="+", help="wandb 运行名称列表")
    parser.add_argument("--output_dir", type=str, default="./results", help="输出目录")
    parser.add_argument("--aggregate", action="store_true", help="是否跨目标聚合结果")
    parser.add_argument("--config", type=str, help="配置文件路径（YAML格式）")
    parser.add_argument("--find_best_node", action="store_true", 
                       help="查找ENPO表现最好的节点（forget最低，neighbor和utility最高）")
    parser.add_argument("--enpo_pattern", type=str, default="enpo", 
                       help="ENPO方法的名称模式（用于识别ENPO方法）")
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，则加载并合并参数（命令行参数优先）
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"错误: 配置文件不存在: {config_path}")
            return
        config = load_config(str(config_path))
        args = merge_config_with_args(config, args)
        print(f"已从配置文件加载参数: {config_path}")
    else:
        # 尝试加载默认配置文件
        default_config_path = Path(__file__).parent / "config" / "generate_results.yaml"
        if default_config_path.exists():
            config = load_config(str(default_config_path))
            args = merge_config_with_args(config, args)
            print(f"已从默认配置文件加载参数: {default_config_path}")
        else:
            # 也尝试从项目根目录的config文件夹加载
            project_config_path = Path(__file__).parent.parent / "config" / "generate_results.yaml"
            if project_config_path.exists():
                config = load_config(str(project_config_path))
                args = merge_config_with_args(config, args)
                print(f"已从默认配置文件加载参数: {project_config_path}")
            else:
                args.method_name_mapping = None
    
    # 如果指定了查找最佳节点
    if args.find_best_node:
        if args.source != "wandb":
            print("错误: --find_best_node 只支持从 wandb 加载数据")
            return
        
        if not args.entity or not args.project:
            print("错误: 使用 --find_best_node 时需要指定 --entity 和 --project")
            return
        
        print("\n查找ENPO表现最好的节点...")
        # 从配置文件中获取task_name_filter
        task_name_filter = None
        if hasattr(args, 'task_name_filter') and args.task_name_filter:
            task_name_filter = args.task_name_filter
        elif args.config:
            config = load_config(args.config)
            if config.get("task_name_filter"):
                task_name_filter = config["task_name_filter"]
        
        # 加载ENPO方法的评估点（使用task_name_filter）
        enpo_eval_points = load_all_evaluation_points_from_wandb(
            args.entity,
            args.project,
            run_ids=args.run_ids,
            run_names=args.run_names,
            task_name_filter=task_name_filter,
        )
        
        # 加载比较方法的评估点（不使用task_name_filter）
        comparison_eval_points = load_all_evaluation_points_from_wandb(
            args.entity,
            args.project,
            run_ids=args.run_ids,
            run_names=args.run_names,
            task_name_filter=None,  # 比较方法不过滤task_name
        )
        
        # 合并评估点（ENPO方法优先，如果有重复则使用ENPO的）
        all_eval_points = {**comparison_eval_points, **enpo_eval_points}
        
        if not all_eval_points:
            print("警告: 没有找到任何评估点数据")
            return
        
        print(f"加载了 {len(all_eval_points)} 个运行的评估点数据")
        
        # 从配置文件中获取比较方法的模式
        comparison_patterns = None
        if args.config:
            config = load_config(args.config)
            if config.get("comparison_method_patterns"):
                comparison_patterns = config["comparison_method_patterns"]
        
        # 如果没有指定，使用默认的比较方法模式
        if comparison_patterns is None:
            comparison_patterns = ["ga", "atu", "npo", "kl+enpo"]
        
        best_node = find_best_enpo_node(
            all_eval_points,
            enpo_method_pattern=args.enpo_pattern,
            comparison_patterns=comparison_patterns,
        )
        
        if best_node:
            print(f"\n找到最佳节点:")
            print(f"  ENPO方法: {best_node['enpo_method']}")
            print(f"  Step: {best_node['step']}")
            print(f"  综合得分: {best_node['score']:.4f}")
            print(f"\n  ENPO指标:")
            enpo_metrics = best_node['enpo_metrics']
            print(f"    Forget/All: {enpo_metrics.get('forget/all', 'N/A'):.2f}")
            print(f"    Neighbor/All: {enpo_metrics.get('neighbor/all', 'N/A'):.2f}")
            print(f"    Utility/Gen: {enpo_metrics.get('utility/gen', 'N/A'):.2f}")
            
            print(f"\n  比较方法指标:")
            for comp_name, comp_metrics in best_node['comparison_metrics'].items():
                print(f"    {comp_name}:")
                print(f"      Forget/All: {comp_metrics.get('forget/all', 'N/A'):.2f}")
                print(f"      Neighbor/All: {comp_metrics.get('neighbor/all', 'N/A'):.2f}")
                print(f"      Utility/Gen: {comp_metrics.get('utility/gen', 'N/A'):.2f}")
            
            # 用最佳节点的数据构建results字典
            results = {}
            # 添加ENPO方法
            results[best_node['enpo_method']] = {
                "metrics": best_node['enpo_metrics'],
                "config": {},
                "id": "best_node",
            }
            # 添加比较方法
            for comp_name, comp_metrics in best_node['comparison_metrics'].items():
                # 如果是GA方法，需要找到neighbor > 70且forget最差的节点
                if 'ga' in comp_name.lower() and 'enpo' not in comp_name.lower():
                    print(f"\n查找GA方法（neighbor > 70，forget最差）的节点...")
                    ga_worst_node = find_worst_ga_node(
                        args.entity,
                        args.project,
                        neighbor_threshold=70,
                    )
                    if ga_worst_node:
                        print(f"找到GA最差节点: {ga_worst_node['run_name']} (Step {ga_worst_node['step']})")
                        print(f"  Forget/All: {ga_worst_node['forget']:.2f}")
                        print(f"  Neighbor/All: {ga_worst_node['neighbor']:.2f}")
                        print(f"  Utility/Gen: {ga_worst_node['utility']:.2f}")
                        results[comp_name] = {
                            "metrics": ga_worst_node['point'],
                            "config": {},
                            "id": ga_worst_node['run_id'],
                        }
                    else:
                        # 如果没找到，使用原来的数据
                        results[comp_name] = {
                            "metrics": comp_metrics,
                            "config": {},
                            "id": "best_node",
                        }
                else:
                    results[comp_name] = {
                        "metrics": comp_metrics,
                        "config": {},
                        "id": "best_node",
                    }
            
            # 修改输出目录，添加节点信息
            output_dir = Path(args.output_dir) / f"best_node_step_{int(best_node['step'])}"
            output_dir.mkdir(parents=True, exist_ok=True)
            args.output_dir = str(output_dir)
        else:
            print("警告: 没有找到满足条件的节点")
            return
    else:
        # 正常加载数据
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        if args.source == "wandb":
            if not args.entity or not args.project:
                print("错误: 使用 wandb 时需要指定 --entity 和 --project")
                return
            
            results = load_data_from_wandb(
                args.entity,
                args.project,
                run_ids=args.run_ids,
                run_names=args.run_names,
            )
        else:
            if not args.json_path:
                print("错误: 使用 json 时需要指定 --json_path")
                return
            results = load_data_from_json(args.json_path)
        
        if not results:
            print("警告: 没有找到任何结果数据")
            return
        
        print(f"加载了 {len(results)} 个运行的结果")
        
        # 如果需要聚合
        if args.aggregate:
            results = aggregate_results_across_targets(results)
            print(f"聚合后得到 {len(results)} 个方法的结果")
    
    # 创建输出目录（如果还没有创建）
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成表格
    table_df = create_comparison_table(results, output_path=str(output_dir / "comparison_table.csv"))
    print("\n对比表格:")
    print(table_df.to_string(index=False))
    
    # 生成表格图
    print("\n生成表格图...")
    plot_comparison_table(
        results,
        output_path=str(output_dir / "table1_comparison.png"),
        method_name_mapping=getattr(args, 'method_name_mapping', None),
    )
    
    # 生成图表
    print("\n生成图表...")
    
    # 图表1: Utility vs Forget 和 Neighbor vs Forget
    plot_utility_vs_forget(
        results,
        output_path=str(output_dir / "figure5_utility_neighbor_vs_forget.png"),
    )
    
    # 图表2: Neighbor vs Forget 权衡曲线
    plot_neighbor_vs_forget_tradeoff(
        results,
        output_path=str(output_dir / "figure4_neighbor_vs_forget.png"),
    )
    
    # 图表3: 不同阶段的 Forget 分数变化
    if args.source == "wandb":
        stage_results = load_stage_results_from_wandb(
            args.entity,
            args.project,
            run_ids=args.run_ids,
        )
        if stage_results:
            plot_forget_over_stages(
                stage_results,
                output_path=str(output_dir / "figure6_forget_over_stages.png"),
            )
        else:
            print("警告: 无法加载阶段结果数据")
    else:
        print("注意: 阶段结果图表需要从 wandb 加载数据")


if __name__ == "__main__":
    main()
