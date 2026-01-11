"""
快速开始示例
Quick Start Example
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os

# 添加算法模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithms.regression import MultipleLinearRegressionModel
from algorithms.ensemble import RandomForestModel


def example_1_linear_regression():
    """
    示例1：多元线性回归
    """
    print("=" * 60)
    print("示例1：多元线性回归")
    print("=" * 60)
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    y = X @ true_coef + np.random.randn(n_samples) * 0.5
    
    # 创建DataFrame
    feature_names = [f'特征{i+1}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = MultipleLinearRegressionModel(normalize=True)
    model.fit(X_train, y_train)
    
    # 评估
    metrics = model.evaluate(X_test, y_test)
    print("\n模型性能:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 查看系数
    print("\n模型系数:")
    print(model.get_coefficients())
    
    # 交叉验证
    cv_results = model.cross_validate(X_train, y_train, cv=5)
    print(f"\n交叉验证 R²: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
    
    print("\n✓ 示例1完成\n")


def example_2_random_forest():
    """
    示例2：随机森林回归
    """
    print("=" * 60)
    print("示例2：随机森林回归")
    print("=" * 60)
    
    # 生成示例数据（非线性关系）
    np.random.seed(42)
    n_samples = 500
    n_features = 8
    
    X = np.random.randn(n_samples, n_features)
    # 复杂的非线性关系
    y = (X[:, 0]**2 + 2*X[:, 1] - X[:, 2]**3 + 
         0.5*X[:, 3]*X[:, 4] + np.random.randn(n_samples) * 0.5)
    
    feature_names = [f'特征{i+1}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = RandomForestModel(
        task_type='regression',
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 评估
    metrics = model.evaluate(X_test, y_test)
    print("\n模型性能:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 特征重要性
    print("\n特征重要性（前5个）:")
    importance = model.get_feature_importance(top_n=5)
    print(importance)
    
    # 交叉验证
    cv_results = model.cross_validate(X_train, y_train, cv=5)
    print(f"\n交叉验证 R²: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
    
    print("\n✓ 示例2完成\n")


def example_3_classification():
    """
    示例3：分类任务
    """
    print("=" * 60)
    print("示例3：随机森林分类")
    print("=" * 60)
    
    from sklearn.datasets import make_classification
    
    # 生成分类数据
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=10,
        n_redundant=5,
        random_state=42
    )
    
    feature_names = [f'特征{i+1}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = RandomForestModel(
        task_type='classification',
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 评估
    metrics = model.evaluate(X_test, y_test)
    print("\n模型性能:")
    print(f"  准确率: {metrics['Accuracy']:.4f}")
    print("\n分类报告:")
    print(metrics['Classification Report'])
    
    # 特征重要性
    print("\n特征重要性（前5个）:")
    importance = model.get_feature_importance(top_n=5)
    print(importance)
    
    print("\n✓ 示例3完成\n")


def example_4_preprocessing():
    """
    示例4：数据预处理
    """
    print("=" * 60)
    print("示例4：数据预处理")
    print("=" * 60)
    
    from algorithms.utils import DataPreprocessor
    
    # 创建带缺失值的数据
    np.random.seed(42)
    n_samples = 200
    
    data = pd.DataFrame({
        '数值特征1': np.random.randn(n_samples),
        '数值特征2': np.random.randn(n_samples) * 10 + 50,
        '类别特征': np.random.choice(['A', 'B', 'C'], n_samples),
        '目标变量': np.random.randn(n_samples)
    })
    
    # 插入缺失值
    data.loc[np.random.choice(data.index, 20, replace=False), '数值特征1'] = np.nan
    data.loc[np.random.choice(data.index, 10, replace=False), '类别特征'] = np.nan
    
    print("\n原始数据缺失值:")
    print(data.isnull().sum())
    
    # 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 处理缺失值
    data_filled = preprocessor.handle_missing_values(data, strategy='mean')
    print("\n处理后缺失值:")
    print(data_filled.isnull().sum())
    
    # 编码分类变量
    data_encoded = preprocessor.encode_categorical(data_filled, method='onehot')
    print(f"\n编码后数据形状: {data_encoded.shape}")
    print(f"列名: {data_encoded.columns.tolist()}")
    
    # 特征缩放
    numeric_cols = data_encoded.select_dtypes(include=[np.number]).columns.drop('目标变量')
    data_scaled = preprocessor.scale_features(
        data_encoded,
        method='standard',
        columns=numeric_cols.tolist()
    )
    
    print("\n前5行处理后的数据:")
    print(data_scaled.head())
    
    print("\n✓ 示例4完成\n")


def main():
    """
    运行所有示例
    """
    print("\n" + "=" * 60)
    print("美赛机器学习算法模板 - 快速开始示例")
    print("MCM ML Templates - Quick Start Examples")
    print("=" * 60 + "\n")
    
    # 运行所有示例
    example_1_linear_regression()
    example_2_random_forest()
    example_3_classification()
    example_4_preprocessing()
    
    print("=" * 60)
    print("所有示例运行完成！")
    print("查看 complete_workflow_example.py 了解完整建模流程")
    print("=" * 60)


if __name__ == "__main__":
    main()
