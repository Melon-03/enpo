# 比较分析结果使用说明

`compare_analysis_results.py` 用于将多个模型的分析结果绘制在同一张图上进行比较。

## 基本用法

### 1. 比较两个模型的结果

```bash
python project/tasks/compare_analysis_results.py \
  --results results/hidden_states_analysis/model1/analysis_results.json \
           results/hidden_states_analysis/model2/analysis_results.json \
  --output results/comparison
```

### 2. 指定要比较的问题和关键词

```bash
python project/tasks/compare_analysis_results.py \
  --results results/hidden_states_analysis/npo_1_Stephen_King/analysis_results.json \
           results/hidden_states_analysis/microsoft_Phi-3-mini-4k-instruct/analysis_results.json \
  --question "Who wrote The Sun Dog?" \
  --keywords stephen sun sorry \
  --output results/comparison \
  --format pdf
```

### 3. 比较所有问题（不指定 --question）

如果不指定 `--question`，程序会比较所有问题，为每个问题生成一张图：

```bash
python project/tasks/compare_analysis_results.py \
  --results results/hidden_states_analysis/*/analysis_results.json \
  --output results/comparison
```

### 4. 比较所有关键词（不指定 --keywords）

如果不指定 `--keywords`，程序会比较所有关键词：

```bash
python project/tasks/compare_analysis_results.py \
  --results results/hidden_states_analysis/model1/analysis_results.json \
           results/hidden_states_analysis/model2/analysis_results.json \
  --question "Who wrote The Sun Dog?" \
  --output results/comparison
```

## 参数说明

- `--results`: **必需**，分析结果JSON文件路径列表（可以指定多个）
- `--output`: 输出目录（默认：`results/comparison`）
- `--question`: 要比较的问题（如果不指定，会比较所有问题）
- `--keywords`: 要比较的关键词列表（如果不指定，会比较所有关键词）
- `--labels`: 模型标签映射文件（JSON格式，用于自定义模型显示名称）
- `--format`: 输出图片格式，可选 `png`、`pdf`、`svg`（默认：`png`，推荐使用 `pdf`）

## 自定义模型标签

可以创建一个JSON文件来定义模型的显示名称，使图表更易读：

**创建 `model_labels.json`：**
```json
{
  "npo_1_Stephen_King": "Unlearned Model",
  "microsoft_Phi-3-mini-4k-instruct": "Base Model",
  "1_Stephen_King": "Fine-tuned Model"
}
```

**使用标签文件：**
```bash
python project/tasks/compare_analysis_results.py \
  --results results/hidden_states_analysis/*/analysis_results.json \
  --labels model_labels.json \
  --output results/comparison
```

## 实际示例

### 示例1：比较unlearned模型和base模型

```bash
python project/tasks/compare_analysis_results.py \
  --results results/hidden_states_analysis/npo_1_Stephen_King/analysis_results.json \
           results/hidden_states_analysis/microsoft_Phi-3-mini-4k-instruct/analysis_results.json \
  --question "Who wrote The Sun Dog?" \
  --keywords stephen sun \
  --output results/comparison \
  --format pdf
```

### 示例2：比较多个模型的所有问题

```bash
python project/tasks/compare_analysis_results.py \
  --results results/hidden_states_analysis/npo_1_Stephen_King/analysis_results.json \
           results/hidden_states_analysis/microsoft_Phi-3-mini-4k-instruct/analysis_results.json \
           results/hidden_states_analysis/1_Stephen_King/analysis_results.json \
  --keywords stephen sun sorry \
  --output results/comparison \
  --format pdf
```

### 示例3：使用通配符批量比较

```bash
# 比较所有模型的结果
python project/tasks/compare_analysis_results.py \
  --results $(find results/hidden_states_analysis -name "analysis_results.json") \
  --question "Who wrote The Sun Dog?" \
  --output results/comparison \
  --format pdf
```

## 输出说明

- 结果会保存到指定的输出目录
- 文件名格式：`comparison_{问题}.{格式}`
- 例如：`comparison_Who_wrote_The_Sun_Dog.pdf`

## 图表说明

- **不同模型**：使用不同颜色区分
- **不同关键词**：使用不同标记（marker）和线型（linestyle）区分
- **标签格式**：`模型名_关键词`
- **Y轴**：自动调整范围，小数值使用科学计数法显示

## 注意事项

1. **格式推荐**：由于matplotlib后端问题，推荐使用 `--format pdf` 而不是 `png`
2. **文件路径**：确保所有指定的JSON文件路径都存在
3. **数据一致性**：不同模型的结果应该有相同的问题和关键词（程序会自动处理缺失的情况）

## 查看帮助

```bash
python project/tasks/compare_analysis_results.py --help
```
