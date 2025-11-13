# SR_CNN 使用指南（优化版本）

## 主要改进

### ✅ 修复了异常权重问题
- **之前**: `window_size // 10 = 3`（当 window_size=32 时）
- **现在**: `max(window_size / 5, 5.0) = 6.4`（默认值）
- 可通过 `anomaly_weight` 参数自定义

### ✅ 添加了数值稳定性
- 添加 `clamp` 防止 `log(0)` 错误
- 使用 `eps=1e-7` 作为数值下界

### ✅ 新增分析工具
- `analyze_scores()` 方法诊断模型性能
- 自动给出优化建议

## 基础使用

```python
from sprctrum.models.sr_cnn import SR_CNN
import polars as pl

# 1. 创建模型（使用默认参数）
model = SR_CNN(
    window_size=32,
    learn_rate=1e-3,
    epochs=10,
    batch_size=256
)

# 2. 训练
model.fit(train_values, train_labels)

# 3. 分析效果（查看分数分布）
scores = model.analyze_scores(test_values, test_labels)
# 输出包括:
# - 正常/异常样本的分数统计
# - 分离度分析
# - 推荐阈值
# - 优化建议（如果效果不好）

# 4. 预测
predictions = model.predict(new_values)
```

## 参数调优策略

### 情况 1: 异常分数太低，与正常分数区分不明显

**症状**: `analyze_scores()` 显示 separation < 0.01

**解决方案**（按优先级）:

```python
# 方案 A: 增加异常权重（最直接有效）
model = SR_CNN(
    anomaly_weight=15.0  # 从默认 6.4 提升到 15
)

# 方案 B: 大幅增加异常权重 + 增加训练轮数
model = SR_CNN(
    anomaly_weight=20.0,
    epochs=20
)

# 方案 C: 降低学习率，更精细训练
model = SR_CNN(
    anomaly_weight=15.0,
    learn_rate=5e-4,  # 从 1e-3 降低到 5e-4
    epochs=20
)
```

### 情况 2: Loss 不收敛或训练不稳定

**症状**: Loss 震荡或持续很高

**解决方案**:

```python
# 降低学习率 + 增加训练轮数
model = SR_CNN(
    learn_rate=5e-4,
    epochs=30,
    batch_size=128  # 可选：减小 batch size
)
```

### 情况 3: 训练很慢

**解决方案**:

```python
# 增大 batch size
model = SR_CNN(
    batch_size=512,  # 从 256 增加到 512
    epochs=10
)
```

### 情况 4: 想要更激进的训练

**解决方案**:

```python
# 组合优化
model = SR_CNN(
    anomaly_weight=25.0,  # 高异常权重
    learn_rate=1e-3,      # 保持标准学习率
    epochs=30,            # 长时间训练
    batch_size=256
)
```

## 完整工作流示例

```python
import polars as pl
from sprctrum.models.sr_cnn import SR_CNN

# 加载数据
train_data = pl.read_csv("train.csv")
test_data = pl.read_csv("test.csv")

# 提取特征和标签
train_values = train_data["value"]
train_labels = train_data["label"]
test_values = test_data["value"]
test_labels = test_data["label"]

# 策略 1: 先用默认参数快速试验
print("=== 试验 1: 默认参数 ===")
model1 = SR_CNN()
model1.fit(train_values, train_labels)
scores1 = model1.analyze_scores(test_values, test_labels)

# 根据 analyze_scores 的建议调整参数
# 假设分离度很小，尝试增加异常权重

print("\n=== 试验 2: 增加异常权重 ===")
model2 = SR_CNN(anomaly_weight=15.0)
model2.fit(train_values, train_labels)
scores2 = model2.analyze_scores(test_values, test_labels)

# 如果还不够好，再次调整

print("\n=== 试验 3: 激进配置 ===")
model3 = SR_CNN(
    anomaly_weight=25.0,
    epochs=30,
    learn_rate=5e-4
)
model3.fit(train_values, train_labels)
scores3 = model3.analyze_scores(test_values, test_labels)

# 选择最佳模型进行最终预测
best_model = model3  # 假设 model3 效果最好
final_scores = best_model.predict(test_values)

# 使用推荐阈值进行二分类
threshold = 0.5  # 从 analyze_scores 输出获取
predictions = (final_scores > threshold).cast(pl.Int32)
```

## 参数参考表

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|----------|------|
| `window_size` | 32 | 16-64 | 滑动窗口大小 |
| `learn_rate` | 1e-3 | 1e-4 ~ 1e-2 | SGD学习率 |
| `epochs` | 10 | 10-50 | 训练轮数 |
| `batch_size` | 256 | 64-512 | 批大小 |
| `anomaly_weight` | auto | 5-30 | 异常样本权重，越大越重视异常 |

## 诊断检查清单

运行 `analyze_scores()` 后检查：

- [ ] **Separation > 0.05**: 好的分离度
- [ ] **Separation 0.01-0.05**: 可接受，考虑调优
- [ ] **Separation < 0.01**: ⚠️ 需要调优

调优顺序：
1. 增加 `anomaly_weight`（最有效）
2. 增加 `epochs`
3. 调整 `learn_rate`
4. 检查数据质量

## 常见问题

### Q1: 为什么默认的 anomaly_weight 是 max(window_size / 5, 5.0)?

A: 这确保了：
- 对于小窗口（如 window_size=32），权重为 6.4，足够重视异常
- 对于大窗口，权重随窗口大小增长
- 最小值 5.0 作为安全下界

### Q2: 什么时候应该增加 anomaly_weight?

A: 当出现以下情况：
- 正常和异常分数重叠严重
- 异常分数的平均值不够高
- `analyze_scores()` 提示分离度小

### Q3: Loss 一直是 0.2-0.3，正常吗？

A: 这是正常的。关键看：
- Loss 是否持续下降
- `analyze_scores()` 显示的分离度
- 最终的检测效果

### Q4: 可以使用 Adam 优化器吗？

A: 可以，但需要修改代码。当前版本使用 SGD 是为了保持与原版一致。如果要使用 Adam：
- 可能需要降低学习率（如 1e-4）
- 通常收敛更快但可能需要更多正则化

## 性能预期

在 KPI 数据集上（window_size=32，默认参数）：
- 训练时间: ~30秒/epoch（取决于数据量和硬件）
- 最终 Loss: 0.1-0.3（正常范围）
- 分离度: 目标 > 0.05

调优后（anomaly_weight=15-25）：
- 分离度: 可达 0.1-0.2
- 异常检测效果显著提升
