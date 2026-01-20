#!/usr/bin/env python
"""
从 wandb 导出实验数据到 JSON 文件
用于后续分析和可视化
"""

import argparse
import json
import wandb
from pathlib import Path


def export_wandb_data(
    entity: str,
    project: str,
    output_path: str,
    run_ids: list = None,
    run_names: list = None,
    include_history: bool = False,
):
    """
    从 wandb 导出数据
    
    Args:
        entity: wandb entity
        project: wandb project name
        output_path: 输出 JSON 文件路径
        run_ids: 可选的运行 ID 列表
        run_names: 可选的运行名称列表
        include_history: 是否包含历史数据（所有阶段）
    """
    import pandas as pd  # 如果包含历史数据需要 pandas
    
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    
    results = {}
    
    for run in runs:
        # 过滤运行
        if run_ids and run.id not in run_ids:
            continue
        if run_names and run.name not in run_names:
            continue
        
        print(f"处理运行: {run.name} ({run.id})")
        
        # 获取历史数据
        history = run.history()
        
        # 提取最后一个阶段的结果
        final_metrics = {}
        if not history.empty:
            for key in history.columns:
                if key.startswith("report/"):
                    values = history[key].dropna()
                    if len(values) > 0:
                        final_metrics[key.replace("report/", "")] = float(values.iloc[-1])
        
        # 也从摘要中获取（可能包含最新的结果）
        if hasattr(run, 'summary'):
            for key, value in run.summary.items():
                if key.startswith("report/"):
                    metric_key = key.replace("report/", "")
                    if metric_key not in final_metrics:
                        try:
                            final_metrics[metric_key] = float(value)
                        except (ValueError, TypeError):
                            pass
        
        # 获取配置
        config = {}
        if hasattr(run, 'config'):
            config = dict(run.config)
        
        run_data = {
            "metrics": final_metrics,
            "config": config,
            "id": run.id,
            "name": run.name,
        }
        
        # 如果包含历史数据
        if include_history:
            history_data = []
            for idx, row in history.iterrows():
                stage_data = {}
                for key in row.index:
                    if key.startswith("report/"):
                        value = row[key]
                        if not pd.isna(value):
                            stage_data[key.replace("report/", "")] = float(value)
                if stage_data:
                    stage_data["step"] = int(idx)
                    if "report/stage_number" in row.index and not pd.isna(row["report/stage_number"]):
                        stage_data["stage_number"] = int(row["report/stage_number"])
                    history_data.append(stage_data)
            run_data["history"] = history_data
        
        results[run.name] = run_data
    
    # 保存到文件
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n成功导出 {len(results)} 个运行的数据到: {output_path}")
    
    # 打印摘要
    print("\n导出的运行:")
    for name, data in results.items():
        metrics_count = len(data["metrics"])
        print(f"  - {name}: {metrics_count} 个指标")
        if include_history and "history" in data:
            print(f"    (包含 {len(data['history'])} 个历史记录)")


def main():
    parser = argparse.ArgumentParser(description="从 wandb 导出实验数据")
    parser.add_argument("--entity", type=str, required=True, help="wandb entity")
    parser.add_argument("--project", type=str, required=True, help="wandb project name")
    parser.add_argument("--output", type=str, default="wandb_export.json", help="输出 JSON 文件路径")
    parser.add_argument("--run_ids", type=str, nargs="+", help="运行 ID 列表")
    parser.add_argument("--run_names", type=str, nargs="+", help="运行名称列表")
    parser.add_argument("--include_history", action="store_true", help="包含历史数据（所有阶段）")
    
    args = parser.parse_args()
    
    export_wandb_data(
        entity=args.entity,
        project=args.project,
        output_path=args.output,
        run_ids=args.run_ids,
        run_names=args.run_names,
        include_history=args.include_history,
    )


if __name__ == "__main__":
    main()
