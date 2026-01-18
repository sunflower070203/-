"""
美赛数据预处理Python模板
MCM/ICM Data Preprocessing Python Templates

本模板提供10种常用数据预处理方法：
1. 标准化 (Standardization)
2. 归一化 (Normalization)
3. 主成分分析 (PCA)
4. 标签编码 (Label Encoding)
5. 独热编码 (One Hot Encoding)
6. 过采样 (Over Sampling)
7. 滑动窗口 (Sliding Window)
8. 插值 (Interpolation)
9. 降采样 (Under-sampling)
10. 特征选择 (Feature Selection)

作者：美赛机器学习算法模板
日期：2024
"""

import numpy as np
import pandas as pd
import sys
import os

# 添加算法模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithms.utils import DataPreprocessor
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


def template_1_standardization():
    """
    模板1: 标准化 (Standardization)
    
    用途：将特征缩放到均值为0，方差为1
    适用场景：
    - 需要消除不同量纲影响
    - 算法对特征尺度敏感（如SVM、神经网络、KNN）
    - 数据服从正态分布或近似正态分布
    """
    print("\n" + "=" * 80)
    print("模板1: 标准化 (Standardization)")
    print("=" * 80)
    
    # 创建示例数据
    data = pd.DataFrame({
        '身高(cm)': [165, 170, 175, 180, 185],
        '体重(kg)': [55, 60, 65, 70, 75],
        '年龄': [20, 25, 30, 35, 40]
    })
    
    print("\n原始数据:")
    print(data)
    print(f"\n数据统计:\n{data.describe()}")
    
    # 标准化
    preprocessor = DataPreprocessor()
    data_standardized = preprocessor.scale_features(data, method='standard')
    
    print("\n标准化后数据:")
    print(data_standardized)
    print(f"\n标准化后统计:\n{data_standardized.describe()}")
    
    print("\n✓ 标准化后均值≈0，标准差≈1")
    return data_standardized


def template_2_normalization():
    """
    模板2: 归一化 (Normalization)
    
    用途：将特征缩放到[0, 1]范围
    适用场景：
    - 需要将数据限制在特定范围
    - 神经网络输入层
    - 图像处理
    - 不假设数据分布
    """
    print("\n" + "=" * 80)
    print("模板2: 归一化 (Normalization)")
    print("=" * 80)
    
    # 创建示例数据
    data = pd.DataFrame({
        '温度(°C)': [15, 20, 25, 30, 35],
        '湿度(%)': [40, 50, 60, 70, 80],
        '风速(m/s)': [2, 5, 8, 11, 14]
    })
    
    print("\n原始数据:")
    print(data)
    
    # 归一化（Min-Max缩放）
    preprocessor = DataPreprocessor()
    data_normalized = preprocessor.scale_features(data, method='minmax')
    
    print("\n归一化后数据:")
    print(data_normalized)
    print(f"\n归一化后统计:\n{data_normalized.describe()}")
    
    print("\n✓ 归一化后所有值在[0, 1]范围内")
    return data_normalized


def template_3_pca():
    """
    模板3: 主成分分析 (PCA)
    
    用途：降维、去除特征间相关性、提取主要信息
    适用场景：
    - 特征维度过高
    - 特征间存在多重共线性
    - 数据可视化（降至2-3维）
    - 去噪
    """
    print("\n" + "=" * 80)
    print("模板3: 主成分分析 (PCA)")
    print("=" * 80)
    
    # 创建高维示例数据
    np.random.seed(42)
    n_samples = 100
    data = pd.DataFrame({
        f'特征{i+1}': np.random.randn(n_samples) for i in range(10)
    })
    
    # 添加一些相关性
    data['特征2'] = data['特征1'] * 0.8 + np.random.randn(n_samples) * 0.2
    data['特征3'] = data['特征1'] * 0.6 + np.random.randn(n_samples) * 0.4
    
    print(f"\n原始数据形状: {data.shape}")
    print(f"原始数据前5行:\n{data.head()}")
    
    # PCA降维
    preprocessor = DataPreprocessor()
    data_pca = preprocessor.apply_pca(data, variance_threshold=0.95)
    
    print(f"\nPCA后数据形状: {data_pca.shape}")
    print(f"PCA后数据前5行:\n{data_pca.head()}")
    
    if preprocessor.pca:
        print(f"\n各主成分解释方差比例:")
        for i, ratio in enumerate(preprocessor.pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
        print(f"\n累计解释方差: {preprocessor.pca.explained_variance_ratio_.sum():.4f}")
    
    print("\n✓ PCA成功降维，保留了95%以上的信息")
    return data_pca


def template_4_label_encoding():
    """
    模板4: 标签编码 (Label Encoding)
    
    用途：将分类变量转换为数值
    适用场景：
    - 有序分类变量（如：低、中、高）
    - 树模型（决策树、随机森林）的分类特征
    - 目标变量编码
    """
    print("\n" + "=" * 80)
    print("模板4: 标签编码 (Label Encoding)")
    print("=" * 80)
    
    # 创建示例数据
    data = pd.DataFrame({
        '城市': ['北京', '上海', '广州', '深圳', '北京', '上海'],
        '教育程度': ['本科', '硕士', '博士', '本科', '硕士', '博士'],
        '收入等级': ['低', '中', '高', '中', '高', '低'],
        '年龄': [25, 30, 35, 28, 32, 27]
    })
    
    print("\n原始数据:")
    print(data)
    
    # 标签编码
    preprocessor = DataPreprocessor()
    categorical_cols = ['城市', '教育程度', '收入等级']
    data_encoded = preprocessor.encode_categorical(
        data, 
        method='label', 
        columns=categorical_cols
    )
    
    print("\n标签编码后数据:")
    print(data_encoded)
    
    print("\n编码映射:")
    for col in categorical_cols:
        if col in preprocessor.encoders:
            encoder = preprocessor.encoders[col]
            print(f"{col}: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
    
    print("\n✓ 分类变量已转换为数值标签")
    return data_encoded


def template_5_onehot_encoding():
    """
    模板5: 独热编码 (One Hot Encoding)
    
    用途：将分类变量转换为二进制向量
    适用场景：
    - 无序分类变量
    - 线性模型（线性回归、逻辑回归）
    - 神经网络
    - 避免引入虚假的顺序关系
    """
    print("\n" + "=" * 80)
    print("模板5: 独热编码 (One Hot Encoding)")
    print("=" * 80)
    
    # 创建示例数据
    data = pd.DataFrame({
        '颜色': ['红', '绿', '蓝', '红', '绿'],
        '尺寸': ['S', 'M', 'L', 'M', 'S'],
        '价格': [10, 20, 30, 15, 12]
    })
    
    print("\n原始数据:")
    print(data)
    print(f"原始数据形状: {data.shape}")
    
    # 独热编码
    preprocessor = DataPreprocessor()
    data_onehot = preprocessor.encode_categorical(
        data, 
        method='onehot', 
        columns=['颜色', '尺寸']
    )
    
    print("\n独热编码后数据:")
    print(data_onehot)
    print(f"编码后数据形状: {data_onehot.shape}")
    print(f"新列名: {data_onehot.columns.tolist()}")
    
    print("\n✓ 分类变量已转换为独热编码向量")
    print("注意: 使用drop_first=True避免多重共线性")
    return data_onehot


def template_6_oversampling():
    """
    模板6: 过采样 (Over Sampling)
    
    用途：增加少数类样本，平衡数据集
    适用场景：
    - 分类问题中类别严重不平衡
    - 少数类样本非常重要（如欺诈检测）
    - 有足够的计算资源处理更多数据
    """
    print("\n" + "=" * 80)
    print("模板6: 过采样 (Over Sampling)")
    print("=" * 80)
    
    # 创建不平衡的分类数据
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        weights=[0.9, 0.1],  # 90% vs 10%
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f'特征{i+1}' for i in range(5)])
    
    print("\n原始数据集:")
    print(f"总样本数: {len(y)}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  类别{cls}: {count} ({count/len(y)*100:.1f}%)")
    
    # 过采样
    preprocessor = DataPreprocessor()
    X_resampled, y_resampled = preprocessor.oversample(X_df, y)
    
    print(f"\n过采样后数据集:")
    print(f"总样本数: {len(y_resampled)}")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  类别{cls}: {count} ({count/len(y_resampled)*100:.1f}%)")
    
    print("\n✓ 数据集已平衡，各类别样本数相等")
    return X_resampled, y_resampled


def template_7_sliding_window():
    """
    模板7: 滑动窗口 (Sliding Window)
    
    用途：将时间序列转换为监督学习问题
    适用场景：
    - 时间序列预测
    - 序列数据建模
    - LSTM/RNN输入准备
    - 股票价格预测、天气预测等
    """
    print("\n" + "=" * 80)
    print("模板7: 滑动窗口 (Sliding Window)")
    print("=" * 80)
    
    # 创建时间序列数据
    np.random.seed(42)
    n_points = 100
    time = pd.date_range('2024-01-01', periods=n_points, freq='D')
    
    # 模拟温度数据：季节性 + 趋势 + 噪声
    seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, n_points))
    trend = np.linspace(0, 5, n_points)
    noise = np.random.randn(n_points) * 2
    temperature = 20 + seasonal + trend + noise
    
    ts_data = pd.DataFrame({
        'date': time,
        'temperature': temperature
    })
    
    print("\n原始时间序列数据:")
    print(ts_data.head(10))
    print(f"数据点数: {len(ts_data)}")
    
    # 创建滑动窗口
    preprocessor = DataPreprocessor()
    window_size = 7  # 使用过去7天预测下一天
    X, y = preprocessor.create_sliding_window(
        ts_data[['temperature']].values,
        window_size=window_size,
        step=1
    )
    
    print(f"\n滑动窗口结果:")
    print(f"特征矩阵形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    print(f"\n示例 - 第1个窗口:")
    print(f"输入特征(过去{window_size}天): {X[0]}")
    print(f"目标值(第{window_size+1}天): {y[0]:.2f}")
    
    print("\n✓ 时间序列已转换为监督学习格式")
    print(f"每个样本使用过去{window_size}个时间步预测下一个时间步")
    return X, y


def template_8_interpolation():
    """
    模板8: 插值 (Interpolation)
    
    用途：填充时间序列或有序数据中的缺失值
    适用场景：
    - 时间序列数据缺失
    - 传感器数据缺失
    - 需要保持数据平滑性
    - 等间隔采样数据
    """
    print("\n" + "=" * 80)
    print("模板8: 插值 (Interpolation)")
    print("=" * 80)
    
    # 创建带缺失值的时间序列
    np.random.seed(42)
    n_points = 50
    time = pd.date_range('2024-01-01', periods=n_points, freq='H')
    
    # 生成平滑曲线
    values = np.sin(np.linspace(0, 4*np.pi, n_points)) * 10 + 50
    
    ts_data = pd.DataFrame({
        'time': time,
        'value': values
    })
    
    # 随机插入缺失值
    missing_indices = np.random.choice(ts_data.index, 10, replace=False)
    ts_data.loc[missing_indices, 'value'] = np.nan
    
    print("\n原始数据（带缺失值）:")
    print(ts_data.head(15))
    print(f"缺失值数量: {ts_data['value'].isnull().sum()}")
    
    # 线性插值
    preprocessor = DataPreprocessor()
    ts_interpolated = preprocessor.interpolate_missing(
        ts_data, 
        method='linear',
        columns=['value']
    )
    
    print("\n插值后数据:")
    print(ts_interpolated.head(15))
    print(f"缺失值数量: {ts_interpolated['value'].isnull().sum()}")
    
    print("\n✓ 缺失值已通过插值填充")
    print("可选方法: 'linear', 'polynomial', 'spline', 'time'")
    return ts_interpolated


def template_9_undersampling():
    """
    模板9: 降采样 (Under-sampling)
    
    用途：减少多数类样本，平衡数据集
    适用场景：
    - 分类问题中类别严重不平衡
    - 数据量很大，需要减少训练时间
    - 计算资源有限
    """
    print("\n" + "=" * 80)
    print("模板9: 降采样 (Under-sampling)")
    print("=" * 80)
    
    # 创建不平衡的分类数据
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        weights=[0.85, 0.15],  # 85% vs 15%
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f'特征{i+1}' for i in range(5)])
    
    print("\n原始数据集:")
    print(f"总样本数: {len(y)}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  类别{cls}: {count} ({count/len(y)*100:.1f}%)")
    
    # 降采样
    preprocessor = DataPreprocessor()
    X_resampled, y_resampled = preprocessor.undersample(X_df, y)
    
    print(f"\n降采样后数据集:")
    print(f"总样本数: {len(y_resampled)}")
    unique, counts = np.unique(y_resampled, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  类别{cls}: {count} ({count/len(y_resampled)*100:.1f}%)")
    
    print("\n✓ 数据集已平衡，各类别样本数相等")
    print("注意: 降采样会丢失部分信息，适合大数据集")
    return X_resampled, y_resampled


def template_10_feature_selection():
    """
    模板10: 特征选择 (Feature Selection)
    
    用途：选择最重要的特征，去除冗余特征
    适用场景：
    - 特征维度过高
    - 存在无关或冗余特征
    - 提高模型性能和可解释性
    - 减少过拟合风险
    """
    print("\n" + "=" * 80)
    print("模板10: 特征选择 (Feature Selection)")
    print("=" * 80)
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 200
    
    # 生成特征：部分有用，部分冗余
    X = np.random.randn(n_samples, 10)
    
    # 目标变量只与部分特征相关
    y = 2*X[:, 0] + 3*X[:, 2] - 1.5*X[:, 5] + np.random.randn(n_samples) * 0.5
    
    X_df = pd.DataFrame(X, columns=[f'特征{i+1}' for i in range(10)])
    
    print(f"\n原始数据:")
    print(f"特征数量: {X_df.shape[1]}")
    print(f"样本数量: {X_df.shape[0]}")
    print(f"\n前5行特征数据:\n{X_df.head()}")
    
    # 特征选择
    preprocessor = DataPreprocessor()
    X_selected, selected_features = preprocessor.select_features(
        X_df, y,
        k=5,  # 选择前5个最重要的特征
        task_type='regression',
        method='f_test'
    )
    
    print(f"\n特征选择结果:")
    print(f"选择的特征: {selected_features}")
    print(f"选择后数据形状: {X_selected.shape}")
    
    if isinstance(X_selected, pd.DataFrame):
        print(f"\n选择后数据前5行:\n{X_selected.head()}")
    
    print("\n✓ 已选择最重要的特征")
    print("可选方法: 'f_test' (F检验), 'mutual_info' (互信息)")
    return X_selected, selected_features


def comprehensive_example():
    """
    综合示例：完整的数据预处理流程
    """
    print("\n" + "=" * 80)
    print("综合示例：完整的数据预处理流程")
    print("=" * 80)
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 500
    
    data = pd.DataFrame({
        '年龄': np.random.randint(20, 60, n_samples),
        '收入(万元)': np.random.randint(5, 50, n_samples),
        '工作年限': np.random.randint(0, 30, n_samples),
        '教育程度': np.random.choice(['本科', '硕士', '博士'], n_samples),
        '城市': np.random.choice(['北京', '上海', '广州', '深圳'], n_samples),
        '信用评分': np.random.randint(300, 850, n_samples)
    })
    
    # 插入缺失值
    data.loc[np.random.choice(data.index, 30, replace=False), '收入(万元)'] = np.nan
    data.loc[np.random.choice(data.index, 20, replace=False), '教育程度'] = np.nan
    
    print(f"\n步骤1: 原始数据")
    print(f"数据形状: {data.shape}")
    print(f"缺失值:\n{data.isnull().sum()}")
    print(f"\n前5行:\n{data.head()}")
    
    preprocessor = DataPreprocessor()
    
    # 步骤2: 处理缺失值（插值）
    print(f"\n步骤2: 处理缺失值")
    data = preprocessor.handle_missing_values(data, strategy='mean')
    print(f"✓ 缺失值已处理")
    
    # 步骤3: 编码分类变量
    print(f"\n步骤3: 编码分类变量")
    categorical_cols = ['教育程度', '城市']
    data = preprocessor.encode_categorical(data, method='onehot', columns=categorical_cols)
    print(f"✓ 分类变量已编码，数据形状: {data.shape}")
    
    # 步骤4: 特征缩放（标准化）
    print(f"\n步骤4: 特征标准化")
    numeric_cols = [col for col in data.columns if col != '信用评分']
    data_scaled = data.copy()
    data_scaled[numeric_cols] = preprocessor.scale_features(
        data[numeric_cols], 
        method='standard'
    )[numeric_cols]
    print(f"✓ 特征已标准化")
    
    # 步骤5: 特征选择
    print(f"\n步骤5: 特征选择")
    X = data_scaled.drop('信用评分', axis=1)
    y = data_scaled['信用评分']
    X_selected, selected_features = preprocessor.select_features(
        X, y, k=5, task_type='regression'
    )
    print(f"✓ 选择了{len(selected_features)}个最重要的特征")
    
    # 步骤6: PCA降维
    print(f"\n步骤6: PCA降维")
    X_pca = preprocessor.apply_pca(X_selected, variance_threshold=0.95)
    print(f"✓ PCA降维完成")
    
    print("\n" + "=" * 80)
    print("数据预处理流程完成！")
    print(f"最终数据形状: {X_pca.shape}")
    print("=" * 80)
    
    return X_pca, y


def main():
    """
    运行所有模板示例
    """
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  美赛数据预处理Python模板 - 完整示例集".center(76) + "*")
    print("*" + "  MCM/ICM Data Preprocessing Templates - Complete Examples".center(76) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    
    # 运行所有模板
    template_1_standardization()
    template_2_normalization()
    template_3_pca()
    template_4_label_encoding()
    template_5_onehot_encoding()
    template_6_oversampling()
    template_7_sliding_window()
    template_8_interpolation()
    template_9_undersampling()
    template_10_feature_selection()
    
    # 综合示例
    comprehensive_example()
    
    print("\n" + "=" * 80)
    print("所有模板示例运行完成！")
    print("=" * 80)
    print("\n使用说明:")
    print("1. 根据您的具体问题选择合适的预处理方法")
    print("2. 标准化/归一化：选择其一，取决于数据分布和算法")
    print("3. PCA：适用于高维数据降维")
    print("4. 编码：标签编码用于有序分类，独热编码用于无序分类")
    print("5. 采样：过采样和降采样用于解决类别不平衡")
    print("6. 滑动窗口：专用于时间序列预测")
    print("7. 插值：用于填充时间序列或有序数据的缺失值")
    print("8. 特征选择：减少特征维度，提高模型性能")
    print("\n美赛建议:")
    print("- 在论文中详细说明每个预处理步骤的原因")
    print("- 对比不同预处理方法的效果")
    print("- 使用可视化展示预处理前后的数据变化")
    print("=" * 80)


if __name__ == "__main__":
    main()
