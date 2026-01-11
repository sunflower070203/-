"""
Multiple Linear Regression Template
用于美赛C/E/F题的多元线性回归模板
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


class MultipleLinearRegressionModel:
    """
    多元线性回归模型类
    适用于连续变量预测问题
    """
    
    def __init__(self, normalize=True):
        """
        初始化模型
        
        Parameters:
        -----------
        normalize : bool, default=True
            是否对特征进行标准化
        """
        self.model = LinearRegression()
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.feature_names = None
        
    def fit(self, X, y):
        """
        训练模型
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            训练特征
        y : array-like, shape (n_samples,)
            目标变量
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        if self.normalize:
            X = self.scaler.fit_transform(X)
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        预测
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            测试特征
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            预测值
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.normalize:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        评估模型
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            测试特征
        y : array-like, shape (n_samples,)
            真实值
            
        Returns:
        --------
        metrics : dict
            包含MSE, RMSE, MAE, R2等评估指标
        """
        y_pred = self.predict(X)
        
        metrics = {
            'MSE': mean_squared_error(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MAE': mean_absolute_error(y, y_pred),
            'R2': r2_score(y, y_pred)
        }
        
        return metrics
    
    def get_coefficients(self):
        """
        获取回归系数
        
        Returns:
        --------
        coef_df : DataFrame
            特征名称和对应系数
        """
        if self.feature_names:
            return pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': self.model.coef_
            })
        else:
            return self.model.coef_
    
    def cross_validate(self, X, y, cv=5):
        """
        交叉验证
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            特征
        y : array-like, shape (n_samples,)
            目标变量
        cv : int, default=5
            交叉验证折数
            
        Returns:
        --------
        scores : dict
            交叉验证分数
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.normalize:
            X = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.model, X, y, cv=cv, 
                                scoring='r2')
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def plot_predictions(self, y_true, y_pred, title='Predictions vs Actual'):
        """
        绘制预测值vs实际值散点图
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
        title : str
            图表标题
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, y_true, y_pred):
        """
        绘制残差图
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 残差散点图
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot')
        
        # 残差直方图
        axes[1].hist(residuals, bins=30, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        
        plt.tight_layout()
        plt.show()


# 使用示例
def example_usage():
    """
    使用示例
    """
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    y = X @ true_coef + np.random.randn(n_samples) * 0.5
    
    # 创建DataFrame
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = MultipleLinearRegressionModel(normalize=True)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    metrics = model.evaluate(X_test, y_test)
    print("Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 获取系数
    print("\nModel Coefficients:")
    print(model.get_coefficients())
    
    # 交叉验证
    cv_results = model.cross_validate(X_train, y_train, cv=5)
    print(f"\nCross-validation R2: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")
    
    return model, X_test, y_test, y_pred


if __name__ == "__main__":
    model, X_test, y_test, y_pred = example_usage()
