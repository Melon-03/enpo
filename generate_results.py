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
        
        # 提取最后一个 unlearning 阶段的评估结果
        eval_data = {}
        for key in history.columns:
            if key.startswith("report/"):
                # 获取最后一个非 NaN 值
                values = history[key].dropna()
                if len(values) > 0:
                    eval_data[key.replace("report/", "")] = values.iloc[-1]
        
        # 也尝试从摘要中获取
        if hasattr(run, 'summary'):
            for key, value in run.summary.items():
                if key.startswith("report/") and key.replace("report/", "") not in eval_data:
                    eval_data[key.replace("report/", "")] = value
        
        # 获取运行配置信息
        config = run.config if hasattr(run, 'config') else {}
        
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
    
    for method, metrics in results.items():
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
    
    for method, metrics in results.items():
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
    
    for method, metrics in results.items():
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
    
    args = parser.parse_args()
    
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
    
    # 生成表格
    table_df = create_comparison_table(results, output_path=str(output_dir / "comparison_table.csv"))
    print("\n对比表格:")
    print(table_df.to_string(index=False))
    
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
