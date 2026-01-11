"""
Data Preprocessing Utilities
用于美赛C/E/F题的数据预处理工具
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   LabelEncoder, OneHotEncoder)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (SelectKBest, f_regression, f_classif,
                                       mutual_info_regression, mutual_info_classif)
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessor:
    """
    数据预处理类
    提供数据清洗、特征缩放、编码等功能
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
    
    def handle_missing_values(self, df, strategy='mean', fill_value=None):
        """
        处理缺失值
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        strategy : str, default='mean'
            填充策略: 'mean', 'median', 'most_frequent', 'constant'
        fill_value : any, default=None
            当strategy='constant'时使用的填充值
            
        Returns:
        --------
        df_filled : DataFrame
            填充后的数据
        """
        df_filled = df.copy()
        
        # 数值列
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            if strategy == 'constant':
                imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            else:
                imputer = SimpleImputer(strategy=strategy)
            
            df_filled[numeric_cols] = imputer.fit_transform(df_filled[numeric_cols])
            self.imputers['numeric'] = imputer
        
        # 分类列
        categorical_cols = df_filled.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df_filled[categorical_cols] = imputer.fit_transform(df_filled[categorical_cols])
            self.imputers['categorical'] = imputer
        
        return df_filled
    
    def scale_features(self, df, method='standard', columns=None):
        """
        特征缩放
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        method : str, default='standard'
            缩放方法: 'standard', 'minmax', 'robust'
        columns : list, default=None
            要缩放的列名列表，None表示所有数值列
            
        Returns:
        --------
        df_scaled : DataFrame
            缩放后的数据
        """
        df_scaled = df.copy()
        
        if columns is None:
            columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        self.scalers[method] = scaler
        
        return df_scaled
    
    def encode_categorical(self, df, method='onehot', columns=None):
        """
        分类变量编码
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        method : str, default='onehot'
            编码方法: 'onehot', 'label'
        columns : list, default=None
            要编码的列名列表，None表示所有分类列
            
        Returns:
        --------
        df_encoded : DataFrame
            编码后的数据
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        if method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=columns, drop_first=True)
        elif method == 'label':
            for col in columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.encoders[col] = le
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        return df_encoded
    
    def remove_outliers(self, df, columns=None, method='iqr', threshold=1.5):
        """
        移除异常值
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        columns : list, default=None
            要检查异常值的列名列表
        method : str, default='iqr'
            检测方法: 'iqr' 或 'zscore'
        threshold : float, default=1.5
            阈值参数
            
        Returns:
        --------
        df_clean : DataFrame
            移除异常值后的数据
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'iqr':
            for col in columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & 
                                   (df_clean[col] <= upper_bound)]
        
        elif method == 'zscore':
            for col in columns:
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean = df_clean[z_scores < threshold]
        
        return df_clean
    
    def select_features(self, X, y, k=10, task_type='regression', method='f_test'):
        """
        特征选择
        
        Parameters:
        -----------
        X : DataFrame or array
            特征数据
        y : array
            目标变量
        k : int, default=10
            选择的特征数量
        task_type : str, default='regression'
            任务类型: 'regression' 或 'classification'
        method : str, default='f_test'
            选择方法: 'f_test' 或 'mutual_info'
            
        Returns:
        --------
        X_selected : DataFrame or array
            选择后的特征
        selected_features : list
            选择的特征名称
        """
        if task_type == 'regression':
            if method == 'f_test':
                score_func = f_regression
            else:
                score_func = mutual_info_regression
        else:
            if method == 'f_test':
                score_func = f_classif
            else:
                score_func = mutual_info_classif
        
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        if isinstance(X, pd.DataFrame):
            selected_features = X.columns[selector.get_support()].tolist()
            X_selected = pd.DataFrame(X_selected, columns=selected_features)
        else:
            selected_features = selector.get_support(indices=True)
        
        return X_selected, selected_features


class DataExplorer:
    """
    数据探索类
    提供数据可视化和统计分析功能
    """
    
    @staticmethod
    def summary_statistics(df):
        """
        生成描述性统计摘要
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
            
        Returns:
        --------
        summary : DataFrame
            统计摘要
        """
        return df.describe()
    
    @staticmethod
    def plot_distributions(df, columns=None, figsize=(15, 10)):
        """
        绘制特征分布图
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        columns : list, default=None
            要绘制的列名列表
        figsize : tuple, default=(15, 10)
            图形大小
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = 3
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, col in enumerate(columns):
            df[col].hist(bins=30, ax=axes[i], edgecolor='black')
            axes[i].set_title(col)
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
        
        # 隐藏多余的子图
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(df, figsize=(12, 10)):
        """
        绘制相关性矩阵热图
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        figsize : tuple, default=(12, 10)
            图形大小
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_missing_values(df, figsize=(10, 6)):
        """
        绘制缺失值分布图
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        figsize : tuple, default=(10, 6)
            图形大小
        """
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            print("No missing values found!")
            return
        
        plt.figure(figsize=figsize)
        missing.plot(kind='bar')
        plt.xlabel('Columns')
        plt.ylabel('Number of Missing Values')
        plt.title('Missing Values Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


# 使用示例
def example_usage():
    """
    使用示例
    """
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples) * 10 + 50,
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.randn(n_samples),
        'target': np.random.randn(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 随机插入缺失值
    df.loc[np.random.choice(df.index, 50, replace=False), 'feature1'] = np.nan
    df.loc[np.random.choice(df.index, 30, replace=False), 'feature3'] = np.nan
    
    print("原始数据:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    print(f"缺失值数量:\n{df.isnull().sum()}")
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    
    # 处理缺失值
    df_filled = preprocessor.handle_missing_values(df, strategy='mean')
    print(f"\n处理缺失值后:\n{df_filled.isnull().sum()}")
    
    # 编码分类变量
    df_encoded = preprocessor.encode_categorical(df_filled, method='onehot')
    print(f"\n编码后数据形状: {df_encoded.shape}")
    
    # 特征缩放
    df_scaled = preprocessor.scale_features(df_encoded, method='standard')
    print(f"\n缩放后数据:")
    print(df_scaled.head())
    
    return df, df_scaled


if __name__ == "__main__":
    df_original, df_processed = example_usage()
