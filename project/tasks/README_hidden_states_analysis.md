# Hidden States Analysis - 模型遗忘分析工具

这个工具用于分析已经unlearn的模型，检查特定知识是否在每一层被遗忘。

## 功能说明

1. **提取各层hidden states**: 通过hook机制提取模型每一层的hidden states
2. **计算Reciprocal Rank**: 对于给定的问题，计算每个关键词在每一层的Reciprocal Rank
3. **可视化结果**: 绘制Reciprocal Rank vs Layer的图表，类似论文中的示意图

## 使用方法

### 1. 配置模型路径

编辑 `config/hidden_states_analysis.yaml` 文件，设置：

```yaml
model_path: "path/to/your/unlearned/model"  # 修改为实际的模型路径
```

### 2. 配置问题和关键词

在配置文件中设置要测试的问题和关键词：

```yaml
questions:
  - "Who wrote The Sun Dog?"
  - "Tell me about the author of The Sun Dog."
  - "Who writes for The Sun?"

keywords:
  - "stephen"  # 应该被注意到的关键词（在输入中查找）
  - "sun"      # 应该被注意到的关键词（在输入中查找）

output_keywords:
  - "sorry"    # 输出关键词，当模型不知道相关知识时会输出这个token
```

### 3. 运行分析

```bash
python project/tasks/hidden_states_analysis.py --config config/hidden_states_analysis.yaml
```

## 输出结果

程序会在 `results/hidden_states_analysis/` 目录下生成：

1. **图表文件**: 每个问题对应一个图表文件，文件名格式为 `rr_vs_layer_{question}.png`
2. **数据文件**: `analysis_results.json` 包含所有原始数据

## 工作原理

1. **Hidden States提取**: 使用PyTorch的hook机制，在模型forward过程中提取每一层的hidden states

2. **Reciprocal Rank计算**:
   - **输入关键词**（如"stephen", "sun"）:
     - 对于每个问题，使用最后一个token的hidden state作为query
     - 计算query与序列中所有位置的cosine相似度
     - 找到关键词token在序列中的位置
     - 计算关键词位置的相似度在所有位置中的排名
     - Reciprocal Rank = 1 / rank
   - **输出关键词**（如"sorry"）:
     - 使用query位置的hidden state通过lm_head计算logits
     - 找到输出关键词token的logit值
     - 计算该logit在所有vocab tokens中的排名
     - Reciprocal Rank = 1 / rank

3. **可视化**: 
   - X轴：Layer (0到N-1)
   - Y轴：Reciprocal Rank (0.0到1.0)
   - 每个关键词一条线

## 示例解读

根据图片示例：
- **"stephen"**: 如果模型有知识，应该在某一层开始注意到"stephen"关键词（RR > 0）
- **"sun"**: 如果模型有知识，应该在某一层注意到"sun"关键词
- **"sorry"**: 这是输出关键词，表示模型不知道相关知识时会输出"sorry"这个token
  - 如果模型真的unlearn了知识，应该在某一层开始倾向于输出"sorry"（RR > 0）
  - 如果模型还有知识，应该不会输出"sorry"（RR = 0）

## 注意事项

1. 每次只加载一个模型进行分析
2. 确保模型路径正确且模型已完全加载
3. 关键词匹配支持子词tokenization
4. 如果模型层数很多，分析可能需要一些时间

## 配置选项说明

- `model_path`: 模型路径（必需）
- `model_type`: 模型类型（可选，默认"llama"）
- `questions`: 要测试的问题列表
- `keywords`: 要检查的关键词列表
- `max_new_tokens`: 生成的最大token数（当前未使用，预留）
- `save_dir`: 结果保存目录
- `save_plot`: 是否保存图表
- `plot_format`: 图表格式（png, pdf, svg）
- `save_data`: 是否保存原始数据为JSON
