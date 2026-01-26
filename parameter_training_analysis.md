# NPO/ENPO/ATU 参数训练方式分析

## 1. NPO (Negative Preference Optimization)

### 代码位置
`project/tasks/unlearning_npo.py`

### 参数设置
```python
# 第164-165行
for param in self.model_theta.parameters():
    param.requires_grad = True

# 第249-256行 - 优化器配置
def configure_optimizers(self):
    return [
        torch.optim.AdamW(
            self.model_theta.parameters(),  # ← 所有参数！
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        ),
    ]
```

### 结论
✅ **全参数微调 (Full Fine-tuning)**
- 优化 `model_theta` 的所有参数（3.8B 参数）
- 没有参数冻结或选择性训练

---

## 2. ENPO (Embedding-based Negative Preference Optimization)

### 代码位置
`project/tasks/unlearning_enpo.py`

### 参数设置
```python
# 第740-752行 - 优化器配置
def configure_optimizers(self):
    return [
        torch.optim.Adam(
            self.embedding_prediction_model.parameters(),  # 小模型
            lr=self.hparams.training_lr,
            weight_decay=self.hparams.training_weight_decay,
        ),
        torch.optim.SGD(
            self.pre_trained_llm.parameters(),  # ← 所有参数！
            lr=self.hparams.unlearning_lr,
            weight_decay=self.hparams.unlearning_weight_decay,
        ),
    ]
```

### 训练阶段
- **Training阶段**: 只训练 `embedding_prediction_model`（小模型）
- **Unlearning阶段**: 训练 `pre_trained_llm` 的所有参数（3.8B 参数）

### 结论
✅ **Unlearning阶段是全参数微调**
- 在unlearning阶段，优化 `pre_trained_llm.parameters()` - 所有参数
- 没有参数冻结或LoRA等参数高效方法

---

## 3. ATU (Align-Then-Unlearn)

### 代码位置
`project/tasks/unlearning_atu.py`

### 参数设置
```python
# 第369-381行 - 优化器配置
def configure_optimizers(self):
    return [
        torch.optim.Adam(
            self.embedding_prediction_model.parameters(),  # 小模型
            lr=self.hparams.training_lr,
            weight_decay=self.hparams.training_weight_decay,
        ),
        torch.optim.SGD(
            self.pre_trained_llm.parameters(),  # ← 所有参数！
            lr=self.hparams.unlearning_lr,
            weight_decay=self.hparams.unlearning_weight_decay,
        ),
    ]
```

### 训练阶段
- **Training阶段**: 只训练 `embedding_prediction_model`
- **Unlearning阶段**: 训练 `pre_trained_llm` 的所有参数

### 结论
✅ **Unlearning阶段是全参数微调**
- 在unlearning阶段，优化 `pre_trained_llm.parameters()` - 所有参数

---

## 总结

| 方法 | 训练阶段 | 参数范围 | 是否全参数微调 |
|------|---------|---------|---------------|
| **NPO** | Unlearning | `model_theta` 所有参数 (3.8B) | ✅ 是 |
| **ENPO** | Training | `embedding_prediction_model` 参数 | ❌ 否（小模型） |
| **ENPO** | Unlearning | `pre_trained_llm` 所有参数 (3.8B) | ✅ 是 |
| **ATU** | Training | `embedding_prediction_model` 参数 | ❌ 否（小模型） |
| **ATU** | Unlearning | `pre_trained_llm` 所有参数 (3.8B) | ✅ 是 |

## 显存占用影响

**全参数微调导致：**
1. **优化器状态占用最大**：30.4 GB（AdamW需要为每个参数存储momentum和variance）
2. **梯度占用**：7.6 GB
3. **模型参数**：7.6 GB

**如果使用参数高效方法（如LoRA），可以：**
- 只优化少量参数（例如1-5%的参数）
- 优化器状态从30.4GB降到约0.3-1.5GB
- 梯度从7.6GB降到约0.3-1.5GB
- **总显存节省：约35-40GB！**

## 建议

如果需要减少显存占用，可以考虑：
1. **使用LoRA**：只优化attention层的低秩矩阵
2. **使用8-bit优化器**：减少优化器状态占用
3. **使用DeepSpeed ZeRO**：分片优化器状态
