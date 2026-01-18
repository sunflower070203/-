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
from sklearn.decomposition import PCA
from sklearn.utils import resample
from scipy import interpolate
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
        self.pca = None
    
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
    
    def apply_pca(self, df, n_components=None, variance_threshold=0.95):
        """
        主成分分析 (Principal Component Analysis)
        降维同时保留主要信息
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        n_components : int or None, default=None
            主成分数量，None表示自动选择保留指定方差比例
        variance_threshold : float, default=0.95
            当n_components=None时，保留的累计方差比例
            
        Returns:
        --------
        df_pca : DataFrame
            降维后的数据
        """
        # 只对数值列进行PCA
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X_numeric = df[numeric_cols].values
        
        if n_components is None:
            # 自动选择组件数量
            pca_temp = PCA()
            pca_temp.fit(X_numeric)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
        
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_numeric)
        
        # 创建新的DataFrame
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
        
        print(f"PCA降维: {len(numeric_cols)}维 → {n_components}维")
        print(f"保留方差比例: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        return df_pca
    
    def oversample(self, X, y, strategy='random', random_state=42):
        """
        过采样 (Over Sampling)
        增加少数类样本，解决类别不平衡问题
        
        Parameters:
        -----------
        X : DataFrame or array
            特征数据
        y : array
            目标变量（分类标签）
        strategy : str, default='random'
            过采样策略: 'random' - 随机过采样
        random_state : int, default=42
            随机种子
            
        Returns:
        --------
        X_resampled : DataFrame or array
            过采样后的特征数据
        y_resampled : array
            过采样后的目标变量
        """
        # 统计各类别样本数
        unique_classes, class_counts = np.unique(y, return_counts=True)
        max_count = class_counts.max()
        
        print(f"原始类别分布: {dict(zip(unique_classes, class_counts))}")
        
        X_resampled_list = []
        y_resampled_list = []
        
        for cls in unique_classes:
            # 获取当前类别的索引
            cls_indices = np.where(y == cls)[0]
            
            if isinstance(X, pd.DataFrame):
                X_cls = X.iloc[cls_indices]
            else:
                X_cls = X[cls_indices]
            y_cls = y[cls_indices]
            
            # 如果是少数类，进行过采样
            if len(cls_indices) < max_count:
                if isinstance(X, pd.DataFrame):
                    X_cls_resampled, y_cls_resampled = resample(
                        X_cls, y_cls,
                        n_samples=max_count,
                        replace=True,
                        random_state=random_state
                    )
                else:
                    indices_resampled = resample(
                        cls_indices,
                        n_samples=max_count,
                        replace=True,
                        random_state=random_state
                    )
                    X_cls_resampled = X[indices_resampled]
                    y_cls_resampled = y[indices_resampled]
            else:
                X_cls_resampled = X_cls
                y_cls_resampled = y_cls
            
            X_resampled_list.append(X_cls_resampled)
            y_resampled_list.append(y_cls_resampled)
        
        # 合并所有类别
        if isinstance(X, pd.DataFrame):
            X_resampled = pd.concat(X_resampled_list, axis=0)
        else:
            X_resampled = np.vstack(X_resampled_list)
        y_resampled = np.concatenate(y_resampled_list)
        
        # 打乱顺序
        shuffle_idx = np.random.RandomState(random_state).permutation(len(y_resampled))
        if isinstance(X, pd.DataFrame):
            X_resampled = X_resampled.iloc[shuffle_idx].reset_index(drop=True)
        else:
            X_resampled = X_resampled[shuffle_idx]
        y_resampled = y_resampled[shuffle_idx]
        
        print(f"过采样后样本数: {len(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def undersample(self, X, y, strategy='random', random_state=42):
        """
        降采样 (Under-sampling)
        减少多数类样本，解决类别不平衡问题
        
        Parameters:
        -----------
        X : DataFrame or array
            特征数据
        y : array
            目标变量（分类标签）
        strategy : str, default='random'
            降采样策略: 'random' - 随机降采样
        random_state : int, default=42
            随机种子
            
        Returns:
        --------
        X_resampled : DataFrame or array
            降采样后的特征数据
        y_resampled : array
            降采样后的目标变量
        """
        # 统计各类别样本数
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_count = class_counts.min()
        
        print(f"原始类别分布: {dict(zip(unique_classes, class_counts))}")
        
        X_resampled_list = []
        y_resampled_list = []
        
        for cls in unique_classes:
            # 获取当前类别的索引
            cls_indices = np.where(y == cls)[0]
            
            if isinstance(X, pd.DataFrame):
                X_cls = X.iloc[cls_indices]
            else:
                X_cls = X[cls_indices]
            y_cls = y[cls_indices]
            
            # 降采样到最小类别数量
            if isinstance(X, pd.DataFrame):
                X_cls_resampled, y_cls_resampled = resample(
                    X_cls, y_cls,
                    n_samples=min_count,
                    replace=False,
                    random_state=random_state
                )
            else:
                indices_resampled = resample(
                    cls_indices,
                    n_samples=min_count,
                    replace=False,
                    random_state=random_state
                )
                X_cls_resampled = X[indices_resampled]
                y_cls_resampled = y[indices_resampled]
            
            X_resampled_list.append(X_cls_resampled)
            y_resampled_list.append(y_cls_resampled)
        
        # 合并所有类别
        if isinstance(X, pd.DataFrame):
            X_resampled = pd.concat(X_resampled_list, axis=0)
        else:
            X_resampled = np.vstack(X_resampled_list)
        y_resampled = np.concatenate(y_resampled_list)
        
        # 打乱顺序
        shuffle_idx = np.random.RandomState(random_state).permutation(len(y_resampled))
        if isinstance(X, pd.DataFrame):
            X_resampled = X_resampled.iloc[shuffle_idx].reset_index(drop=True)
        else:
            X_resampled = X_resampled[shuffle_idx]
        y_resampled = y_resampled[shuffle_idx]
        
        print(f"降采样后样本数: {len(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def create_sliding_window(self, data, window_size, target_col=None, step=1):
        """
        滑动窗口 (Sliding Window)
        将时间序列数据转换为监督学习问题
        
        Parameters:
        -----------
        data : DataFrame or array
            时间序列数据
        window_size : int
            窗口大小（历史时间步数）
        target_col : str or int, default=None
            目标列名或索引，None表示使用最后一列
        step : int, default=1
            滑动步长
            
        Returns:
        --------
        X : array
            特征数据（窗口数据）
        y : array
            目标变量（下一时刻的值）
        """
        if isinstance(data, pd.DataFrame):
            if target_col is None:
                target_col = data.columns[-1]
            values = data.values
            target_idx = data.columns.get_loc(target_col)
        else:
            values = data
            target_idx = -1 if target_col is None else target_col
        
        X, y = [], []
        
        for i in range(0, len(values) - window_size, step):
            # 提取窗口数据作为特征
            window = values[i:i + window_size]
            X.append(window.flatten())
            
            # 提取下一时刻的目标值
            target_value = values[i + window_size, target_idx]
            y.append(target_value)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"滑动窗口: 窗口大小={window_size}, 步长={step}")
        print(f"生成样本数: {len(X)}, 特征维度: {X.shape[1]}")
        
        return X, y
    
    def interpolate_missing(self, df, method='linear', columns=None):
        """
        插值 (Interpolation)
        使用插值方法填充缺失值
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        method : str, default='linear'
            插值方法: 'linear', 'polynomial', 'spline', 'time'
        columns : list, default=None
            要插值的列名列表，None表示所有数值列
            
        Returns:
        --------
        df_interpolated : DataFrame
            插值后的数据
        """
        df_interpolated = df.copy()
        
        if columns is None:
            columns = df_interpolated.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if df_interpolated[col].isnull().any():
                if method == 'linear':
                    df_interpolated[col] = df_interpolated[col].interpolate(method='linear')
                elif method == 'polynomial':
                    df_interpolated[col] = df_interpolated[col].interpolate(method='polynomial', order=2)
                elif method == 'spline':
                    df_interpolated[col] = df_interpolated[col].interpolate(method='spline', order=3)
                elif method == 'time':
                    df_interpolated[col] = df_interpolated[col].interpolate(method='time')
                else:
                    # 默认使用线性插值
                    df_interpolated[col] = df_interpolated[col].interpolate(method='linear')
                
                # 对于开头和结尾的NaN，使用前向/后向填充
                df_interpolated[col] = df_interpolated[col].bfill().ffill()
        
        print(f"插值方法: {method}")
        print(f"处理后缺失值: {df_interpolated[columns].isnull().sum().sum()}")
        
        return df_interpolated
    
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
    使用示例 - 展示所有数据预处理功能
    """
    print("=" * 80)
    print("数据预处理工具使用示例")
    print("=" * 80)
    
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
    
    print("\n原始数据:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    print(f"缺失值数量:\n{df.isnull().sum()}")
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    
    # 1. 处理缺失值
    print("\n" + "=" * 80)
    print("1. 处理缺失值")
    print("-" * 80)
    df_filled = preprocessor.handle_missing_values(df, strategy='mean')
    print(f"处理后缺失值:\n{df_filled.isnull().sum()}")
    
    # 2. 编码分类变量
    print("\n" + "=" * 80)
    print("2. 编码分类变量")
    print("-" * 80)
    df_encoded = preprocessor.encode_categorical(df_filled, method='onehot')
    print(f"独热编码后数据形状: {df_encoded.shape}")
    print(f"新列名: {df_encoded.columns.tolist()}")
    
    # 3. 特征缩放（标准化）
    print("\n" + "=" * 80)
    print("3. 特征缩放（标准化）")
    print("-" * 80)
    df_scaled = preprocessor.scale_features(df_encoded, method='standard')
    print(f"标准化后数据统计:\n{df_scaled.describe()}")
    
    # 4. 主成分分析 (PCA)
    print("\n" + "=" * 80)
    print("4. 主成分分析 (PCA)")
    print("-" * 80)
    df_pca = preprocessor.apply_pca(df_scaled, variance_threshold=0.95)
    print(f"PCA后数据形状: {df_pca.shape}")
    if preprocessor.pca:
        print(f"各主成分解释方差比例: {preprocessor.pca.explained_variance_ratio_}")
    
    # 5. 插值示例
    print("\n" + "=" * 80)
    print("5. 插值 (Interpolation)")
    print("-" * 80)
    # 创建带缺失值的时间序列数据
    ts_data = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=100, freq='D'),
        'value': np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.randn(100) * 0.1
    })
    # 随机插入缺失值
    ts_data.loc[np.random.choice(ts_data.index, 10, replace=False), 'value'] = np.nan
    print(f"插值前缺失值数量: {ts_data['value'].isnull().sum()}")
    ts_interpolated = preprocessor.interpolate_missing(ts_data, method='linear', columns=['value'])
    print(f"插值后缺失值数量: {ts_interpolated['value'].isnull().sum()}")
    
    # 6. 滑动窗口示例
    print("\n" + "=" * 80)
    print("6. 滑动窗口 (Sliding Window)")
    print("-" * 80)
    # 创建时间序列数据
    time_series = np.sin(np.linspace(0, 10*np.pi, 200))
    X_window, y_window = preprocessor.create_sliding_window(
        time_series.reshape(-1, 1),
        window_size=10,
        step=1
    )
    print(f"窗口特征形状: {X_window.shape}")
    print(f"目标变量形状: {y_window.shape}")
    
    # 7. 过采样/降采样示例（分类问题）
    print("\n" + "=" * 80)
    print("7. 过采样与降采样")
    print("-" * 80)
    # 创建不平衡的分类数据
    X_imbalanced = np.random.randn(300, 5)
    y_imbalanced = np.array([0]*250 + [1]*50)  # 类别不平衡
    
    print("\n过采样 (Over Sampling):")
    X_over, y_over = preprocessor.oversample(X_imbalanced, y_imbalanced)
    
    print("\n降采样 (Under-sampling):")
    X_under, y_under = preprocessor.undersample(X_imbalanced, y_imbalanced)
    
    # 8. 特征选择
    print("\n" + "=" * 80)
    print("8. 特征选择 (Feature Selection)")
    print("-" * 80)
    X = df_filled.drop(['target', 'feature3'], axis=1)
    y = df_filled['target']
    X_selected, selected_features = preprocessor.select_features(
        X, y, k=2, task_type='regression', method='f_test'
    )
    print(f"选择的特征: {selected_features}")
    print(f"选择后数据形状: {X_selected.shape}")
    
    print("\n" + "=" * 80)
    print("所有预处理示例完成！")
    print("=" * 80)
    
    return df, df_scaled


if __name__ == "__main__":
    df_original, df_processed = example_usage()
