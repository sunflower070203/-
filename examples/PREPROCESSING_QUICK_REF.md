# 美赛数据预处理速查卡
# MCM/ICM Data Preprocessing Quick Reference

## 快速索引 | Quick Index

| 序号 | 中文名称 | 英文名称 | 方法调用 | 使用场景 |
|-----|---------|---------|----------|---------|
| 1 | 标准化 | Standardization | `scale_features(method='standard')` | 正态分布数据、SVM、神经网络 |
| 2 | 归一化 | Normalization | `scale_features(method='minmax')` | 神经网络、图像处理 |
| 3 | 主成分分析 | PCA | `apply_pca(variance_threshold=0.95)` | 高维数据降维、去相关性 |
| 4 | 标签编码 | Label Encoding | `encode_categorical(method='label')` | 有序分类、树模型 |
| 5 | 独热编码 | One Hot Encoding | `encode_categorical(method='onehot')` | 无序分类、线性模型 |
| 6 | 过采样 | Over Sampling | `oversample(X, y)` | 类别不平衡、少数类重要 |
| 7 | 滑动窗口 | Sliding Window | `create_sliding_window(window_size=7)` | 时间序列预测 |
| 8 | 插值 | Interpolation | `interpolate_missing(method='linear')` | 时间序列缺失值 |
| 9 | 降采样 | Under-sampling | `undersample(X, y)` | 类别不平衡、大数据集 |
| 10 | 特征选择 | Feature Selection | `select_features(k=10)` | 高维数据、去冗余 |

---

## 一行代码示例 | One-Line Examples

```python
from algorithms.utils import DataPreprocessor
preprocessor = DataPreprocessor()

# 1. 标准化 Standardization
data_std = preprocessor.scale_features(data, method='standard')

# 2. 归一化 Normalization  
data_norm = preprocessor.scale_features(data, method='minmax')

# 3. PCA降维
data_pca = preprocessor.apply_pca(data, variance_threshold=0.95)

# 4. 标签编码
data_label = preprocessor.encode_categorical(data, method='label', columns=['city'])

# 5. 独热编码
data_onehot = preprocessor.encode_categorical(data, method='onehot', columns=['color'])

# 6. 过采样
X_over, y_over = preprocessor.oversample(X, y)

# 7. 滑动窗口
X_window, y_window = preprocessor.create_sliding_window(ts_data, window_size=7)

# 8. 插值
data_interp = preprocessor.interpolate_missing(data, method='linear')

# 9. 降采样
X_under, y_under = preprocessor.undersample(X, y)

# 10. 特征选择
X_selected, features = preprocessor.select_features(X, y, k=10, task_type='regression')
```

---

## 决策树 | Decision Tree

```
开始数据预处理
│
├─ 有缺失值？
│  ├─ 是 → 时间序列？
│  │      ├─ 是 → 使用插值 (interpolation)
│  │      └─ 否 → 使用填充 (handle_missing_values)
│  └─ 否 → 继续
│
├─ 有分类变量？
│  ├─ 是 → 有序分类？
│  │      ├─ 是 → 标签编码 (label encoding)
│  │      └─ 否 → 独热编码 (one-hot encoding)
│  └─ 否 → 继续
│
├─ 特征尺度差异大？
│  ├─ 是 → 数据正态分布？
│  │      ├─ 是 → 标准化 (standardization)
│  │      └─ 否 → 归一化 (normalization)
│  └─ 否 → 继续
│
├─ 特征维度过高？
│  ├─ 是 → 需要保留原始特征？
│  │      ├─ 是 → 特征选择 (feature selection)
│  │      └─ 否 → PCA降维
│  └─ 否 → 继续
│
├─ 类别不平衡？（分类问题）
│  ├─ 是 → 数据量大？
│  │      ├─ 是 → 降采样 (under-sampling)
│  │      └─ 否 → 过采样 (over-sampling)
│  └─ 否 → 继续
│
└─ 时间序列预测？
   ├─ 是 → 滑动窗口 (sliding window)
   └─ 否 → 完成预处理
```

---

## 常用组合 | Common Combinations

### 组合1: 回归问题标准流程
```python
# 1. 处理缺失值
data = preprocessor.handle_missing_values(data, strategy='mean')

# 2. 编码分类变量
data = preprocessor.encode_categorical(data, method='onehot')

# 3. 标准化
data = preprocessor.scale_features(data, method='standard')

# 4. 特征选择
X_selected, features = preprocessor.select_features(X, y, k=10)
```

### 组合2: 分类问题（类别不平衡）
```python
# 1. 处理缺失值
data = preprocessor.handle_missing_values(data)

# 2. 编码分类变量
data = preprocessor.encode_categorical(data, method='onehot')

# 3. 归一化
data = preprocessor.scale_features(data, method='minmax')

# 4. 过采样
X_resampled, y_resampled = preprocessor.oversample(X, y)

# 5. 特征选择
X_selected, features = preprocessor.select_features(X_resampled, y_resampled, k=10, task_type='classification')
```

### 组合3: 时间序列预测
```python
# 1. 插值填充缺失值
data = preprocessor.interpolate_missing(data, method='linear')

# 2. 标准化
data = preprocessor.scale_features(data, method='standard')

# 3. 滑动窗口
X, y = preprocessor.create_sliding_window(data, window_size=7, step=1)
```

### 组合4: 高维数据处理
```python
# 1. 处理缺失值
data = preprocessor.handle_missing_values(data)

# 2. 编码分类变量
data = preprocessor.encode_categorical(data, method='onehot')

# 3. 标准化
data = preprocessor.scale_features(data, method='standard')

# 4. PCA降维
data_pca = preprocessor.apply_pca(data, variance_threshold=0.95)
```

---

## 参数速查 | Parameter Reference

### scale_features()
- `method`: `'standard'`, `'minmax'`, `'robust'`
- `columns`: 要缩放的列（None=全部数值列）

### apply_pca()
- `n_components`: 主成分数量（None=自动）
- `variance_threshold`: 保留方差比例（默认0.95）

### encode_categorical()
- `method`: `'label'`, `'onehot'`
- `columns`: 要编码的列（None=全部分类列）

### oversample() / undersample()
- `strategy`: `'random'`（随机采样）
- `random_state`: 随机种子（默认42）

### create_sliding_window()
- `window_size`: 窗口大小（历史时间步数）
- `step`: 滑动步长（默认1）
- `target_col`: 目标列名（None=最后一列）

### interpolate_missing()
- `method`: `'linear'`, `'polynomial'`, `'spline'`, `'time'`
- `columns`: 要插值的列（None=全部数值列）

### select_features()
- `k`: 选择的特征数量
- `task_type`: `'regression'`, `'classification'`
- `method`: `'f_test'`, `'mutual_info'`

---

## 美赛常用话术 | Common Phrases for MCM Papers

### 中文版本

1. **标准化**: "为消除不同量纲的影响，我们对所有数值特征进行Z-score标准化处理。"

2. **归一化**: "考虑到神经网络对输入数据范围的敏感性，我们采用Min-Max归一化将特征缩放至[0,1]区间。"

3. **PCA**: "为降低模型复杂度并去除特征间的多重共线性，我们采用主成分分析(PCA)将20个原始特征降至8个主成分，累计解释方差达95.2%。"

4. **编码**: "对于分类变量'城市'和'教育程度'，我们采用独热编码方法，避免引入虚假的顺序关系。"

5. **过采样**: "考虑到原始数据存在严重的类别不平衡问题(90% vs 10%)，我们采用随机过采样技术平衡训练集。"

6. **滑动窗口**: "我们使用7天的历史数据作为滑动窗口，构建时间序列监督学习问题，预测第8天的值。"

7. **插值**: "对于传感器数据中的缺失值，我们采用线性插值方法，保持时间序列的连续性和平滑性。"

8. **特征选择**: "通过F检验方法，我们选择了与目标变量相关性最强的10个特征，既提高了模型性能，又增强了可解释性。"

### English Version

1. **Standardization**: "To eliminate the impact of different scales, we applied Z-score standardization to all numerical features."

2. **Normalization**: "Considering the sensitivity of neural networks to input ranges, we employed Min-Max normalization to scale features to [0,1]."

3. **PCA**: "To reduce model complexity and remove multicollinearity, we applied PCA to reduce 20 features to 8 principal components, retaining 95.2% of variance."

4. **Encoding**: "For categorical variables 'city' and 'education', we used one-hot encoding to avoid introducing artificial ordinal relationships."

5. **Over Sampling**: "Given the severe class imbalance (90% vs 10%), we employed random oversampling to balance the training set."

6. **Sliding Window**: "We used a 7-day sliding window approach to transform the time series into a supervised learning problem."

7. **Interpolation**: "For missing sensor data, we applied linear interpolation to maintain continuity and smoothness of the time series."

8. **Feature Selection**: "Using F-test, we selected the 10 most relevant features, improving both model performance and interpretability."

---

## 调试检查清单 | Debugging Checklist

- [ ] 数据加载成功，无损坏
- [ ] 检查缺失值数量和位置
- [ ] 确认分类变量的类别数量
- [ ] 验证数值特征的分布和范围
- [ ] 检查是否存在异常值
- [ ] 确认目标变量的类型（连续/离散）
- [ ] 验证类别是否平衡（分类问题）
- [ ] 检查特征间相关性
- [ ] 确保时间序列数据有序（如适用）
- [ ] 验证预处理后数据形状正确

---

## 性能优化建议 | Performance Tips

1. **大数据集**: 先采样再预处理，验证流程后再用全量数据
2. **特征过多**: 先粗选特征（方差阈值）再精选（统计检验）
3. **时间序列**: 使用向量化操作代替循环
4. **内存优化**: 使用`dtype`优化（float64→float32）
5. **并行处理**: 使用`n_jobs=-1`参数（如适用）

---

**快速上手**: `python examples/preprocessing_templates.py`
**详细文档**: `examples/PREPROCESSING_GUIDE.md`
**源代码**: `algorithms/utils/preprocessing.py`
