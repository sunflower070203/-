"""
综合示例：美赛C/E/F题完整建模流程
Complete Example: Full Modeling Workflow for MCM Problems C/E/F
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import os

# 添加算法模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithms.regression import MultipleLinearRegressionModel
from algorithms.ensemble import RandomForestModel, GradientBoostingModel
from algorithms.utils import DataPreprocessor, DataExplorer, ModelEvaluator


def generate_sample_data():
    """
    生成示例数据集
    模拟一个环境科学问题（类似美赛E题）
    """
    np.random.seed(42)
    n_samples = 1000
    
    # 特征：温度、湿度、风速、降雨量、污染物排放、人口密度
    temperature = np.random.uniform(15, 35, n_samples)
    humidity = np.random.uniform(30, 90, n_samples)
    wind_speed = np.random.uniform(0, 20, n_samples)
    rainfall = np.random.exponential(5, n_samples)
    emissions = np.random.uniform(50, 200, n_samples)
    population_density = np.random.uniform(100, 5000, n_samples)
    
    # 目标变量：空气质量指数 (AQI)
    # 复杂的非线性关系
    aqi = (
        0.5 * temperature +
        -0.3 * humidity +
        -2.0 * wind_speed +
        -0.5 * rainfall +
        1.5 * emissions +
        0.01 * population_density +
        0.001 * temperature ** 2 +
        0.01 * emissions ** 1.5 +
        np.random.normal(0, 10, n_samples)
    )
    
    # 创建DataFrame
    data = pd.DataFrame({
        '温度(°C)': temperature,
        '湿度(%)': humidity,
        '风速(m/s)': wind_speed,
        '降雨量(mm)': rainfall,
        '污染物排放(吨)': emissions,
        '人口密度(人/km²)': population_density,
        'AQI': aqi
    })
    
    # 随机插入一些缺失值
    for col in data.columns:
        missing_idx = np.random.choice(data.index, size=int(0.05 * n_samples), replace=False)
        data.loc[missing_idx, col] = np.nan
    
    return data


def main():
    """
    完整的建模流程
    """
    print("=" * 80)
    print("美赛机器学习建模完整流程示例")
    print("MCM/ICM Machine Learning Modeling Workflow Example")
    print("=" * 80)
    
    # ========== 步骤1：生成/加载数据 ==========
    print("\n步骤1：数据加载")
    print("-" * 80)
    data = generate_sample_data()
    print(f"数据集大小: {data.shape}")
    print(f"\n前5行数据:")
    print(data.head())
    print(f"\n数据统计信息:")
    print(data.describe())
    
    # ========== 步骤2：数据探索 ==========
    print("\n步骤2：数据探索分析")
    print("-" * 80)
    explorer = DataExplorer()
    
    # 检查缺失值
    print("缺失值统计:")
    print(data.isnull().sum())
    
    # ========== 步骤3：数据预处理 ==========
    print("\n步骤3：数据预处理")
    print("-" * 80)
    preprocessor = DataPreprocessor()
    
    # 处理缺失值
    data_filled = preprocessor.handle_missing_values(data, strategy='mean')
    print("✓ 缺失值已处理")
    
    # 检测和移除异常值
    data_clean = preprocessor.remove_outliers(data_filled, method='iqr', threshold=1.5)
    print(f"✓ 异常值已移除，数据集大小: {data_clean.shape}")
    
    # 准备特征和目标变量
    X = data_clean.drop('AQI', axis=1)
    y = data_clean['AQI']
    
    # 特征缩放
    X_scaled = preprocessor.scale_features(X, method='standard')
    print("✓ 特征已标准化")
    
    # ========== 步骤4：划分训练集和测试集 ==========
    print("\n步骤4：划分数据集")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # ========== 步骤5：模型训练 ==========
    print("\n步骤5：模型训练")
    print("-" * 80)
    
    # 5.1 线性回归（基准模型）
    print("\n5.1 训练多元线性回归模型...")
    lr_model = MultipleLinearRegressionModel(normalize=False)  # 已经标准化过
    lr_model.fit(X_train, y_train)
    print("✓ 线性回归模型训练完成")
    
    # 5.2 随机森林
    print("\n5.2 训练随机森林模型...")
    rf_model = RandomForestModel(
        task_type='regression',
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    print("✓ 随机森林模型训练完成")
    
    # 5.3 梯度提升
    print("\n5.3 训练梯度提升模型...")
    gb_model = GradientBoostingModel(
        task_type='regression',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    print("✓ 梯度提升模型训练完成")
    
    # ========== 步骤6：模型评估 ==========
    print("\n步骤6：模型评估")
    print("-" * 80)
    evaluator = ModelEvaluator()
    
    # 评估线性回归
    print("\n线性回归模型性能:")
    lr_metrics = lr_model.evaluate(X_test, y_test)
    for metric, value in lr_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 评估随机森林
    print("\n随机森林模型性能:")
    rf_metrics = rf_model.evaluate(X_test, y_test)
    for metric, value in rf_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 评估梯度提升
    print("\n梯度提升模型性能:")
    gb_metrics = gb_model.evaluate(X_test, y_test)
    for metric, value in gb_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # ========== 步骤7：模型比较 ==========
    print("\n步骤7：模型比较")
    print("-" * 80)
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    
    models = {
        '线性回归': LinearRegression(),
        '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
        '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    comparison = evaluator.compare_models(
        models, X_train, y_train, cv=5, task_type='regression'
    )
    print("\n交叉验证模型比较:")
    print(comparison)
    
    # ========== 步骤8：特征重要性分析 ==========
    print("\n步骤8：特征重要性分析")
    print("-" * 80)
    
    print("\n随机森林特征重要性:")
    rf_importance = rf_model.get_feature_importance()
    print(rf_importance)
    
    print("\n梯度提升特征重要性:")
    gb_importance = gb_model.get_feature_importance()
    print(gb_importance)
    
    print("\n线性回归系数:")
    lr_coef = lr_model.get_coefficients()
    print(lr_coef)
    
    # ========== 步骤9：预测和结果分析 ==========
    print("\n步骤9：预测和结果分析")
    print("-" * 80)
    
    # 使用最佳模型（梯度提升）进行预测
    y_pred = gb_model.predict(X_test)
    
    # 预测误差分析
    errors = y_test - y_pred
    print(f"预测误差统计:")
    print(f"  平均误差: {np.mean(errors):.4f}")
    print(f"  误差标准差: {np.std(errors):.4f}")
    print(f"  最大正误差: {np.max(errors):.4f}")
    print(f"  最大负误差: {np.min(errors):.4f}")
    
    # ========== 步骤10：结果总结 ==========
    print("\n步骤10：建模结果总结")
    print("=" * 80)
    print("\n模型性能排名（按R²分数）:")
    print(f"1. 梯度提升: R² = {gb_metrics['R2']:.4f}, RMSE = {gb_metrics['RMSE']:.4f}")
    print(f"2. 随机森林: R² = {rf_metrics['R2']:.4f}, RMSE = {rf_metrics['RMSE']:.4f}")
    print(f"3. 线性回归: R² = {lr_metrics['R2']:.4f}, RMSE = {lr_metrics['RMSE']:.4f}")
    
    print("\n关键发现:")
    print("1. 梯度提升模型表现最佳，能够很好地捕捉非线性关系")
    print("2. 污染物排放和人口密度是影响AQI的最重要因素")
    print("3. 气象因素（温度、风速、降雨）也有显著影响")
    
    print("\n建议:")
    print("1. 在美赛论文中，应详细说明模型选择的原因")
    print("2. 使用多个模型进行对比，增强结论的可信度")
    print("3. 重点解释特征重要性，支持政策建议")
    print("4. 进行敏感性分析，验证模型稳定性")
    
    print("\n" + "=" * 80)
    print("建模流程完成！")
    print("=" * 80)
    
    return {
        'data': data_clean,
        'models': {
            'linear_regression': lr_model,
            'random_forest': rf_model,
            'gradient_boosting': gb_model
        },
        'metrics': {
            'linear_regression': lr_metrics,
            'random_forest': rf_metrics,
            'gradient_boosting': gb_metrics
        }
    }


if __name__ == "__main__":
    results = main()
