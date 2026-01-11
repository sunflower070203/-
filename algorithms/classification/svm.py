"""
Support Vector Machine (SVM) Template
用于美赛C/E/F题的支持向量机模板
支持回归和分类任务
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)
import matplotlib.pyplot as plt


class SVMModel:
    """
    支持向量机模型类
    支持回归和分类任务
    """
    
    def __init__(self, task_type='classification', kernel='rbf', 
                 C=1.0, gamma='scale', normalize=True, random_state=42):
        """
        初始化模型
        
        Parameters:
        -----------
        task_type : str, default='classification'
            任务类型: 'regression' 或 'classification'
        kernel : str, default='rbf'
            核函数类型: 'linear', 'poly', 'rbf', 'sigmoid'
        C : float, default=1.0
            正则化参数
        gamma : str or float, default='scale'
            核系数
        normalize : bool, default=True
            是否对特征进行标准化
        random_state : int, default=42
            随机种子
        """
        self.task_type = task_type
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.feature_names = None
        
        if task_type == 'classification':
            self.model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                random_state=random_state,
                probability=True
            )
        else:
            self.model = SVR(
                kernel=kernel,
                C=C,
                gamma=gamma
            )
    
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
    
    def predict_proba(self, X):
        """
        预测概率（仅分类任务）
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            测试特征
            
        Returns:
        --------
        proba : array, shape (n_samples, n_classes)
            预测概率
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.normalize:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
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
            评估指标
        """
        y_pred = self.predict(X)
        
        if self.task_type == 'regression':
            metrics = {
                'MSE': mean_squared_error(y, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                'MAE': mean_absolute_error(y, y_pred),
                'R2': r2_score(y, y_pred)
            }
        else:
            metrics = {
                'Accuracy': accuracy_score(y, y_pred),
                'Classification Report': classification_report(y, y_pred),
                'Confusion Matrix': confusion_matrix(y, y_pred)
            }
        
        return metrics
    
    def tune_hyperparameters(self, X, y, param_grid=None, cv=5):
        """
        超参数调优
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            特征
        y : array-like, shape (n_samples,)
            目标变量
        param_grid : dict, default=None
            超参数搜索空间
        cv : int, default=5
            交叉验证折数
            
        Returns:
        --------
        best_params : dict
            最优超参数
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.normalize:
            X = self.scaler.fit_transform(X)
        
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly']
            }
        
        scoring = 'r2' if self.task_type == 'regression' else 'accuracy'
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, 
            scoring=scoring, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_


# 使用示例
def example_usage_classification():
    """
    分类任务示例
    """
    from sklearn.datasets import make_classification
    
    # 生成示例数据
    X, y = make_classification(n_samples=1000, n_features=20,
                              n_informative=15, n_redundant=5,
                              random_state=42)
    
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = SVMModel(task_type='classification', kernel='rbf', C=1.0)
    model.fit(X_train, y_train)
    
    # 评估
    metrics = model.evaluate(X_test, y_test)
    print("SVM Classification Model Performance:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics['Classification Report'])
    
    return model


def example_usage_regression():
    """
    回归任务示例
    """
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0]**2 + 2*X[:, 1] - X[:, 2]**3 + 
         np.random.randn(n_samples) * 0.5)
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # 训练模型
    model = SVMModel(task_type='regression', kernel='rbf', C=1.0)
    model.fit(X_train, y_train)
    
    # 评估
    metrics = model.evaluate(X_test, y_test)
    print("SVM Regression Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return model


if __name__ == "__main__":
    print("=" * 50)
    print("SVM Classification Example")
    print("=" * 50)
    clf_model = example_usage_classification()
    
    print("\n" + "=" * 50)
    print("SVM Regression Example")
    print("=" * 50)
    reg_model = example_usage_regression()
