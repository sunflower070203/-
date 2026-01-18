# 美赛机器学习算法模板

## 项目简介

本项目为美国大学生数学建模竞赛（MCM/ICM）的C/E/F题提供常用机器学习算法模板，包括多元线性回归、随机森林、集成学习等多种算法的实现和使用示例。

## 目录结构

```
├── algorithms/              # 算法模块
│   ├── regression/         # 回归算法
│   │   └── multiple_linear_regression.py
│   ├── classification/     # 分类算法
│   │   ├── logistic_regression.py
│   │   ├── svm.py
│   │   ├── knn.py
│   │   └── decision_tree.py
│   ├── ensemble/          # 集成学习算法
│   │   ├── random_forest.py
│   │   └── ensemble_methods.py
│   └── utils/             # 工具模块
│       ├── preprocessing.py
│       └── evaluation.py
├── examples/              # 使用示例
├── data/                  # 数据文件夹
├── requirements.txt       # 依赖包
└── README.md             # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 算法列表

### 回归算法

1. **多元线性回归** (`algorithms/regression/multiple_linear_regression.py`)
   - 适用于连续变量预测
   - 支持特征标准化
   - 提供系数解释和可视化

### 分类算法

2. **逻辑回归** (`algorithms/classification/logistic_regression.py`)
   - 二分类和多分类支持
   - 支持L1/L2正则化
   - 提供ROC曲线和混淆矩阵

3. **支持向量机 (SVM)** (`algorithms/classification/svm.py`)
   - 支持多种核函数（线性、RBF、多项式）
   - 回归和分类任务
   - 超参数自动调优

4. **K近邻 (KNN)** (`algorithms/classification/knn.py`)
   - 简单高效的分类回归算法
   - 自动寻找最优K值
   - 支持多种距离度量

5. **决策树** (`algorithms/classification/decision_tree.py`)
   - 可解释性强
   - 树结构可视化
   - 特征重要性分析

### 集成学习算法

6. **随机森林** (`algorithms/ensemble/random_forest.py`)
   - 回归和分类任务
   - 特征重要性排序
   - 超参数调优

7. **梯度提升 (Gradient Boosting)** (`algorithms/ensemble/ensemble_methods.py`)
   - 高精度预测
   - 支持回归和分类
   - 特征重要性分析

8. **AdaBoost** (`algorithms/ensemble/ensemble_methods.py`)
   - 自适应提升算法
   - 减少偏差和方差
   - 适合中小规模数据集

9. **XGBoost** (`algorithms/ensemble/ensemble_methods.py`)
   - 高性能梯度提升
   - 竞赛常用算法
   - 丰富的调参选项

10. **投票集成 (Voting Ensemble)** (`algorithms/ensemble/ensemble_methods.py`)
    - 结合多个模型预测
    - 硬投票和软投票
    - 提高模型稳定性

11. **堆叠集成 (Stacking)** (`algorithms/ensemble/ensemble_methods.py`)
    - 多层模型组合
    - 元学习器优化
    - 最大化模型性能

### 工具模块

12. **数据预处理** (`algorithms/utils/preprocessing.py`)
    - 缺失值处理
    - 特征缩放（标准化、归一化）
    - 主成分分析 (PCA)
    - 分类变量编码（标签编码、独热编码）
    - 过采样 (Over Sampling)
    - 降采样 (Under-sampling)
    - 滑动窗口 (Sliding Window)
    - 插值 (Interpolation)
    - 异常值检测和处理
    - 特征选择
    - **📚 [预处理完整指南](examples/PREPROCESSING_GUIDE.md)**
    - **⚡ [预处理速查卡](examples/PREPROCESSING_QUICK_REF.md)**
    - **💡 [10个模板示例](examples/preprocessing_templates.py)**

13. **模型评估** (`algorithms/utils/evaluation.py`)
    - 回归评估指标（MSE、RMSE、MAE、R²）
    - 分类评估指标（准确率、精确率、召回率、F1分数）
    - 交叉验证
    - 学习曲线和验证曲线
    - 模型比较

## 快速开始

### 1. 多元线性回归示例

```python
from algorithms.regression import MultipleLinearRegressionModel
import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据
data = pd.read_csv('your_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = MultipleLinearRegressionModel(normalize=True)
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
print(metrics)

# 查看系数
print(model.get_coefficients())
```

### 2. 随机森林示例

```python
from algorithms.ensemble import RandomForestModel

# 创建回归模型
model = RandomForestModel(task_type='regression', n_estimators=100)
model.fit(X_train, y_train)

# 评估
metrics = model.evaluate(X_test, y_test)
print(metrics)

# 查看特征重要性
importance = model.get_feature_importance(top_n=10)
print(importance)

# 绘制特征重要性图
model.plot_feature_importance(top_n=20)
```

### 3. 集成学习示例

```python
from algorithms.ensemble import GradientBoostingModel, VotingEnsemble
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 梯度提升
gb_model = GradientBoostingModel(task_type='regression', n_estimators=100)
gb_model.fit(X_train, y_train)

# 投票集成
estimators = [
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('gb', gb_model.model)
]

voting_model = VotingEnsemble(estimators, task_type='regression')
voting_model.fit(X_train, y_train)
metrics = voting_model.evaluate(X_test, y_test)
```

### 4. 数据预处理示例

```python
from algorithms.utils import DataPreprocessor

# 创建预处理器
preprocessor = DataPreprocessor()

# 1. 处理缺失值（插值）
data_filled = preprocessor.interpolate_missing(data, method='linear')

# 2. 编码分类变量
data_encoded = preprocessor.encode_categorical(data_filled, method='onehot')

# 3. 特征缩放（标准化）
data_scaled = preprocessor.scale_features(data_encoded, method='standard')

# 4. PCA降维
data_pca = preprocessor.apply_pca(data_scaled, variance_threshold=0.95)

# 5. 过采样（解决类别不平衡）
X_resampled, y_resampled = preprocessor.oversample(X, y)

# 6. 滑动窗口（时间序列）
X_window, y_window = preprocessor.create_sliding_window(
    time_series_data, 
    window_size=7, 
    step=1
)

# 7. 特征选择
X_selected, selected_features = preprocessor.select_features(
    X, y, k=10, task_type='regression'
)

# 移除异常值
data_clean = preprocessor.remove_outliers(data_scaled, method='iqr', threshold=1.5)

# 数据探索
explorer = DataExplorer()
explorer.plot_correlation_matrix(data)
explorer.plot_distributions(data)
```

**💡 查看完整的10个预处理模板**: `python examples/preprocessing_templates.py`

### 5. 模型评估和比较

```python
from algorithms.utils import ModelEvaluator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 创建评估器
evaluator = ModelEvaluator()

# 比较多个模型
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

comparison = evaluator.compare_models(models, X_train, y_train, cv=5, task_type='regression')
print(comparison)

# 绘制学习曲线
evaluator.plot_learning_curve(models['Random Forest'], X_train, y_train)

# 绘制验证曲线
param_range = [10, 50, 100, 200, 300]
evaluator.plot_validation_curve(
    RandomForestRegressor(random_state=42),
    X_train, y_train,
    param_name='n_estimators',
    param_range=param_range
)
```

## 使用建议

### 美赛C题（数据分析类）

推荐算法：
1. 多元线性回归 - 用于建立基准模型
2. 随机森林 - 处理非线性关系
3. 梯度提升 - 获得更高精度
4. 特征重要性分析 - 解释模型结果

### 美赛E题（环境科学类）

推荐算法：
1. 时间序列分析 + 回归
2. 随机森林 - 处理复杂环境因素
3. 集成学习 - 提高预测稳定性
4. 数据预处理 - 处理缺失和异常值

### 美赛F题（政策类）

推荐算法：
1. 逻辑回归 - 分类和决策分析
2. 决策树 - 可解释的决策规则
3. SVM - 高维数据分类
4. 模型评估 - 验证政策效果

## 注意事项

1. **数据预处理**：使用前务必对数据进行清洗和预处理
2. **特征工程**：根据问题特点构造合适的特征
3. **模型选择**：根据数据规模和问题类型选择合适的算法
4. **超参数调优**：使用交叉验证和网格搜索优化参数
5. **模型解释**：美赛非常重视模型的可解释性
6. **可视化**：使用图表展示分析结果

## 美赛论文建模流程建议

1. **问题分析** → 确定任务类型（回归/分类）
2. **数据探索** → 使用DataExplorer进行初步分析
3. **数据预处理** → 使用DataPreprocessor清洗数据
4. **特征工程** → 构造和选择重要特征
5. **模型训练** → 尝试多个算法并比较
6. **模型优化** → 超参数调优
7. **模型评估** → 使用ModelEvaluator全面评估
8. **结果可视化** → 绘制图表展示结果
9. **敏感性分析** → 验证模型稳定性
10. **撰写论文** → 详细说明建模过程和结果

## 参考文献

- Scikit-learn Documentation: https://scikit-learn.org/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- MCM/ICM Contest Archive: https://www.comap.com/

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请通过GitHub Issues联系。

---

**祝美赛取得好成绩！Good luck with MCM/ICM!**