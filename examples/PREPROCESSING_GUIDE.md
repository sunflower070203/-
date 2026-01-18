# 美赛数据预处理Python模板使用指南
# MCM/ICM Data Preprocessing Templates Usage Guide

## 概述 Overview

本指南详细介绍10种数据预处理技术的使用方法，这些技术是美国大学生数学建模竞赛（MCM/ICM）中最常用的数据预处理方法。

This guide provides detailed instructions for 10 data preprocessing techniques commonly used in MCM/ICM competitions.

## 快速开始 Quick Start

```python
# 安装依赖
pip install -r requirements.txt

# 运行完整示例
python examples/preprocessing_templates.py

# 或直接使用预处理类
from algorithms.utils import DataPreprocessor

preprocessor = DataPreprocessor()
```

## 10种预处理技术详解

### 1. 标准化 (Standardization)

**目的**: 将特征缩放到均值为0，标准差为1

**公式**: `z = (x - μ) / σ`

**适用场景**:
- 算法对特征尺度敏感（SVM、神经网络、KNN）
- 数据服从正态分布
- 需要消除量纲影响

**示例代码**:
```python
preprocessor = DataPreprocessor()
data_standardized = preprocessor.scale_features(data, method='standard')
```

**美赛建议**:
- 在论文中说明为什么选择标准化而不是归一化
- 展示标准化前后的数据统计对比
- 解释标准化如何改善模型性能

---

### 2. 归一化 (Normalization)

**目的**: 将特征缩放到[0, 1]范围

**公式**: `x_norm = (x - x_min) / (x_max - x_min)`

**适用场景**:
- 神经网络输入层
- 图像处理
- 不假设数据分布
- 需要将数据限制在特定范围

**示例代码**:
```python
preprocessor = DataPreprocessor()
data_normalized = preprocessor.scale_features(data, method='minmax')
```

**美赛建议**:
- 说明归一化对异常值敏感的特性
- 在处理异常值后再进行归一化
- 绘制归一化前后的数据分布图

---

### 3. 主成分分析 (PCA)

**目的**: 降维、去除特征间相关性、提取主要信息

**适用场景**:
- 特征维度过高（>10）
- 特征间存在多重共线性
- 数据可视化
- 去除噪声

**示例代码**:
```python
preprocessor = DataPreprocessor()
# 自动选择保留95%方差的主成分数量
data_pca = preprocessor.apply_pca(data, variance_threshold=0.95)

# 或指定主成分数量
data_pca = preprocessor.apply_pca(data, n_components=5)

# 查看解释方差比例
print(preprocessor.pca.explained_variance_ratio_)
```

**美赛建议**:
- 绘制碎石图（Scree Plot）展示方差解释比例
- 说明选择主成分数量的标准
- 解释主要主成分的物理意义
- 展示降维后的2D/3D可视化

---

### 4. 标签编码 (Label Encoding)

**目的**: 将分类变量转换为数值标签

**适用场景**:
- 有序分类变量（低、中、高）
- 树模型（决策树、随机森林）
- 目标变量编码
- 减少内存占用

**示例代码**:
```python
preprocessor = DataPreprocessor()
data_encoded = preprocessor.encode_categorical(
    data, 
    method='label', 
    columns=['education', 'income_level']
)

# 查看编码映射
for col, encoder in preprocessor.encoders.items():
    print(f"{col}: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
```

**美赛建议**:
- 明确说明哪些特征是有序的
- 对于无序特征使用独热编码
- 在表格中展示编码映射关系

---

### 5. 独热编码 (One Hot Encoding)

**目的**: 将分类变量转换为二进制向量

**适用场景**:
- 无序分类变量
- 线性模型（线性回归、逻辑回归）
- 神经网络
- 避免引入虚假的顺序关系

**示例代码**:
```python
preprocessor = DataPreprocessor()
data_onehot = preprocessor.encode_categorical(
    data, 
    method='onehot', 
    columns=['color', 'city']
)
```

**美赛建议**:
- 说明使用drop_first=True避免多重共线性
- 对比独热编码和标签编码的效果
- 注意类别过多时的维度爆炸问题

---

### 6. 过采样 (Over Sampling)

**目的**: 增加少数类样本，平衡数据集

**适用场景**:
- 分类问题中类别严重不平衡
- 少数类样本非常重要（欺诈检测、疾病诊断）
- 有足够计算资源

**示例代码**:
```python
preprocessor = DataPreprocessor()
X_resampled, y_resampled = preprocessor.oversample(X, y, random_state=42)
```

**美赛建议**:
- 展示原始和过采样后的类别分布
- 说明过采样可能导致过拟合
- 考虑使用SMOTE等更高级的过采样方法
- 对比过采样前后的模型性能

---

### 7. 滑动窗口 (Sliding Window)

**目的**: 将时间序列转换为监督学习问题

**适用场景**:
- 时间序列预测
- LSTM/RNN输入准备
- 股票价格预测
- 天气预测
- 传感器数据分析

**示例代码**:
```python
preprocessor = DataPreprocessor()
# 使用过去7个时间步预测下一个时间步
X, y = preprocessor.create_sliding_window(
    time_series_data,
    window_size=7,
    step=1
)
```

**美赛建议**:
- 说明窗口大小的选择依据
- 展示窗口划分的示意图
- 讨论步长对训练样本数量的影响
- 进行窗口大小的敏感性分析

---

### 8. 插值 (Interpolation)

**目的**: 填充时间序列或有序数据中的缺失值

**适用场景**:
- 时间序列数据缺失
- 传感器数据缺失
- 需要保持数据平滑性
- 等间隔采样数据

**示例代码**:
```python
preprocessor = DataPreprocessor()
# 线性插值
data_interpolated = preprocessor.interpolate_missing(
    data, 
    method='linear',
    columns=['temperature', 'humidity']
)

# 其他插值方法
# method='polynomial' - 多项式插值
# method='spline' - 样条插值
# method='time' - 时间加权插值
```

**美赛建议**:
- 对比不同插值方法的效果
- 绘制插值前后的曲线对比图
- 说明插值方法的选择依据
- 讨论插值对数据质量的影响

---

### 9. 降采样 (Under-sampling)

**目的**: 减少多数类样本，平衡数据集

**适用场景**:
- 类别严重不平衡
- 数据量很大，需要减少训练时间
- 计算资源有限
- 多数类包含大量冗余信息

**示例代码**:
```python
preprocessor = DataPreprocessor()
X_resampled, y_resampled = preprocessor.undersample(X, y, random_state=42)
```

**美赛建议**:
- 说明降采样会丢失信息
- 适合大数据集使用
- 考虑使用EasyEnsemble等集成降采样方法
- 对比降采样前后的模型性能

---

### 10. 特征选择 (Feature Selection)

**目的**: 选择最重要的特征，去除冗余特征

**适用场景**:
- 特征维度过高
- 存在无关或冗余特征
- 提高模型性能和可解释性
- 减少过拟合风险

**示例代码**:
```python
preprocessor = DataPreprocessor()
# 回归问题
X_selected, selected_features = preprocessor.select_features(
    X, y, 
    k=10,  # 选择前10个特征
    task_type='regression',
    method='f_test'  # 或 'mutual_info'
)

# 分类问题
X_selected, selected_features = preprocessor.select_features(
    X, y, 
    k=10,
    task_type='classification',
    method='f_test'
)

print(f"选择的特征: {selected_features}")
```

**美赛建议**:
- 绘制特征重要性排序图
- 说明特征选择的标准
- 对比不同k值对模型性能的影响
- 解释为什么某些特征被选中

---

## 完整工作流示例

```python
from algorithms.utils import DataPreprocessor
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 加载数据
data = pd.read_csv('your_data.csv')

# 2. 初始化预处理器
preprocessor = DataPreprocessor()

# 3. 处理缺失值（插值）
data = preprocessor.interpolate_missing(data, method='linear')

# 4. 编码分类变量
categorical_cols = ['city', 'education']
data = preprocessor.encode_categorical(data, method='onehot', columns=categorical_cols)

# 5. 移除异常值
data = preprocessor.remove_outliers(data, method='iqr', threshold=1.5)

# 6. 特征缩放
X = data.drop('target', axis=1)
y = data['target']
X = preprocessor.scale_features(X, method='standard')

# 7. 特征选择
X_selected, selected_features = preprocessor.select_features(
    X, y, k=10, task_type='regression'
)

# 8. PCA降维（可选）
X_pca = preprocessor.apply_pca(X_selected, variance_threshold=0.95)

# 9. 对于不平衡分类问题，进行采样
# X_resampled, y_resampled = preprocessor.oversample(X_pca, y)

# 10. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)
```

## 美赛论文写作建议

### 数据预处理章节结构

```
3. 数据预处理 (Data Preprocessing)

3.1 数据清洗
- 缺失值处理: 使用线性插值填充时间序列缺失值
- 异常值检测: IQR方法识别和处理异常值
- 数据统计: [插入描述性统计表格]

3.2 特征工程
- 分类变量编码: 对城市、教育程度等进行独热编码
- 特征缩放: 标准化所有数值特征
- [插入编码前后对比图]

3.3 降维处理
- PCA分析: 从20个特征降至8个主成分
- 保留方差: 95.2%
- [插入碎石图和主成分载荷矩阵]

3.4 类别平衡
- 问题: 原始数据类别不平衡（90% vs 10%）
- 方法: 随机过采样（SMOTE更优）
- 结果: 平衡后各类别样本数相等
- [插入类别分布对比图]
```

### 可视化建议

1. **数据分布图**: 展示预处理前后的特征分布
2. **相关性热图**: 展示特征间相关性
3. **PCA碎石图**: 展示各主成分解释方差
4. **特征重要性图**: 条形图展示特征重要性排序
5. **类别分布图**: 饼图或柱状图展示类别平衡前后对比

### 表格建议

| 预处理步骤 | 方法 | 参数 | 效果 |
|---------|------|------|------|
| 缺失值处理 | 线性插值 | - | 填充32个缺失值 |
| 异常值处理 | IQR方法 | threshold=1.5 | 移除45个异常样本 |
| 特征编码 | 独热编码 | drop_first=True | 5个分类特征→12个二进制特征 |
| 特征缩放 | 标准化 | - | 均值=0, 标准差=1 |
| 降维 | PCA | variance=0.95 | 20维→8维 |
| 特征选择 | F检验 | k=10 | 选择最重要的10个特征 |

## 常见问题

### Q1: 标准化和归一化如何选择？

**答**: 
- 数据近似正态分布 → 标准化
- 数据分布未知或有界 → 归一化
- SVM、神经网络 → 标准化
- 图像处理 → 归一化

### Q2: 过采样和降采样如何选择？

**答**:
- 数据量小 → 过采样
- 数据量大 → 降采样
- 少数类重要 → 过采样
- 计算资源有限 → 降采样
- 最佳: 结合使用SMOTE和EasyEnsemble

### Q3: PCA会丢失信息吗？

**答**:
- 是的，PCA是有损压缩
- 但可以控制保留方差比例（如95%）
- 丢失的主要是噪声和冗余信息
- 适合特征间高度相关的情况

### Q4: 什么时候需要特征选择？

**答**:
- 特征数量 > 样本数量
- 存在明显无关特征
- 模型训练时间过长
- 需要提高模型可解释性

## 参考资源

- Scikit-learn文档: https://scikit-learn.org/stable/modules/preprocessing.html
- 特征工程最佳实践: [Feature Engineering and Selection](http://www.feat.engineering/)
- 时间序列预处理: [Time Series Analysis](https://otexts.com/fpp3/)

## 许可证

MIT License

---

**祝美赛取得优异成绩！Good luck with MCM/ICM!**
