# 实验结果可视化指南

本指南说明如何生成论文中使用的表格和图表。

## 快速开始

### 从 Wandb 生成结果

```bash
# 基本用法（从 wandb 加载数据）
python generate_results.py \
    --source wandb \
    --entity "1137113080-wuhan-university" \
    --project "atu-unlearning" \
    --output_dir ./results

# 指定特定的运行
python generate_results.py \
    --source wandb \
    --entity "1137113080-wuhan-university" \
    --project "atu-unlearning" \
    --run_names "unlearning-1_Stephen_King" "unlearning-2_Taylor_Swift" \
    --output_dir ./results

# 聚合多个目标的结果（计算平均值）
python generate_results.py \
    --source wandb \
    --entity "1137113080-wuhan-university" \
    --project "atu-unlearning" \
    --aggregate \
    --output_dir ./results
```

### 从 JSON 文件生成结果

首先需要准备 JSON 格式的数据文件：

```json
{
  "method_name_1": {
    "metrics": {
      "forget/fb": 36.3,
      "forget/qa": 40.5,
      "forget/aa": 48.8,
      "forget/all": 42.5,
      "neighbor/qa": 64.4,
      "neighbor/fb": 55.8,
      "neighbor/all": 60.1,
      "utility/gen": 64.2,
      "utility/rea": 58.3,
      "utility/tru": 62.1,
      "utility/fac": 61.5,
      "utility/flu": 63.8
    },
    "config": {
      "method": "ATU (OURS; 50%)"
    }
  },
  "method_name_2": {
    "metrics": {
      "forget/fb": 24.1,
      "forget/qa": 24.8,
      ...
    },
    "config": {
      "method": "ATU (OURS; 35%)"
    }
  }
}
```

然后运行：

```bash
python generate_results.py \
    --source json \
    --json_path results_data.json \
    --output_dir ./results
```

## 生成的文件

脚本会在输出目录中生成以下文件：

1. **comparison_table.csv**: 对比表格（包含所有方法的指标）
2. **figure5_utility_neighbor_vs_forget.png**: 图5 - Utility vs Forget 和 Neighbor vs Forget 权衡图
3. **figure4_neighbor_vs_forget.png**: 图4 - Neighbor vs Forget 权衡曲线
4. **figure6_forget_over_stages.png**: 图6 - 不同阶段 Forget 分数变化

## 表格格式

生成的表格包含以下列：

- **METHOD**: 方法名称
- **FORGET_FB**: Fill-in-the-Blank 遗忘分数（越低越好）
- **FORGET_QA**: Question Answering 遗忘分数（越低越好）
- **FORGET_AA**: Adversarial Attack 遗忘分数（越低越好）
- **FORGET_ALL**: 综合遗忘分数（加权平均，越低越好）
- **NEIGHBOR_QA**: 邻域知识保留 - QA（越高越好）
- **NEIGHBOR_FB**: 邻域知识保留 - FB（越高越好）
- **NEIGHBOR_ALL**: 邻域知识保留 - 综合（越高越好）
- **UTILITY_MMLU**: 模型效用 - MMLU（越高越好）
- **UTILITY_BBH**: 模型效用 - BBH（越高越好）
- **UTILITY_TRUTHFUL**: 模型效用 - TruthfulQA（越高越好）
- **UTILITY_TRIVIA**: 模型效用 - TriviaQA（越高越好）
- **UTILITY_FLUENCY**: 模型效用 - Fluency（越高越好）

## 图表说明

### Figure 4: Neighbor vs Forget Trade-Off
显示遗忘分数和邻域知识保留之间的权衡关系。

### Figure 5: Trade-off between forgetting and model performance
包含两个子图：
- 左图：Utility vs Forget（模型效用 vs 遗忘分数）
- 右图：Neighbor vs Forget（邻域知识保留 vs 遗忘分数）

### Figure 6: Forget scores over stages
显示在不同 unlearning 阶段，Forget 分数的变化趋势。

## 从 Wandb 导出数据

如果需要手动导出 wandb 数据到 JSON，可以使用以下 Python 脚本：

```python
import wandb
import json

api = wandb.Api()
runs = api.runs("your-entity/your-project")

results = {}
for run in runs:
    history = run.history()
    metrics = {}
    
    # 获取最后一个阶段的结果
    for key in history.columns:
        if key.startswith("report/"):
            values = history[key].dropna()
            if len(values) > 0:
                metrics[key.replace("report/", "")] = values.iloc[-1]
    
    results[run.name] = {
        "metrics": metrics,
        "config": dict(run.config) if hasattr(run, 'config') else {},
    }

with open("results_data.json", "w") as f:
    json.dump(results, f, indent=2)
```

## 注意事项

1. **数据格式**: 确保从 wandb 导出的数据包含 `report/` 前缀的指标
2. **阶段数据**: 图6（阶段变化图）需要历史数据，目前只支持从 wandb 加载
3. **层标识**: 如果方法名称中包含层信息（如 "layer_10", "layer_20"），脚本会自动识别并分组
4. **聚合**: 使用 `--aggregate` 选项时，会按 `config.method` 字段分组并计算平均值

## 故障排除

### 找不到数据
- 检查 wandb entity 和 project 名称是否正确
- 确认运行名称或 ID 是否正确
- 检查网络连接（wandb API 需要网络访问）

### 图表不显示
- 确保安装了 matplotlib 和 seaborn: `pip install matplotlib seaborn`
- 检查数据中是否包含必要的指标

### 阶段数据缺失
- 图6需要完整的历史数据，确保 wandb 运行记录了大量历史步骤
- 如果历史数据不完整，图6可能无法生成
