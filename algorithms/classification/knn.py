"""
K-Nearest Neighbors (KNN) Template
用于美赛C/E/F题的K近邻算法模板
支持回归和分类任务
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)
import matplotlib.pyplot as plt


class KNNModel:
    """
    K近邻模型类
    支持回归和分类任务
    """
    
    def __init__(self, task_type='classification', n_neighbors=5, 
                 weights='uniform', metric='minkowski', normalize=True):
        """
        初始化模型
        
        Parameters:
        -----------
        task_type : str, default='classification'
            任务类型: 'regression' 或 'classification'
        n_neighbors : int, default=5
            邻居数量
        weights : str, default='uniform'
            权重函数: 'uniform' 或 'distance'
        metric : str, default='minkowski'
            距离度量: 'euclidean', 'manhattan', 'minkowski'
        normalize : bool, default=True
            是否对特征进行标准化
        """
        self.task_type = task_type
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.feature_names = None
        
        if task_type == 'classification':
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric
            )
        else:
            self.model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric
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
    
    def find_optimal_k(self, X, y, k_range=range(1, 31), cv=5):
        """
        寻找最优K值
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            特征
        y : array-like, shape (n_samples,)
            目标变量
        k_range : range, default=range(1, 31)
            K值范围
        cv : int, default=5
            交叉验证折数
            
        Returns:
        --------
        results : dict
            包含最优K值和交叉验证分数
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.normalize:
            X = self.scaler.fit_transform(X)
        
        scores = []
        for k in k_range:
            if self.task_type == 'classification':
                knn = KNeighborsClassifier(n_neighbors=k)
                scoring = 'accuracy'
            else:
                knn = KNeighborsRegressor(n_neighbors=k)
                scoring = 'r2'
            
            cv_scores = cross_val_score(knn, X, y, cv=cv, scoring=scoring)
            scores.append(cv_scores.mean())
        
        optimal_k = k_range[np.argmax(scores)]
        
        # 更新模型
        self.model.n_neighbors = optimal_k
        
        return {
            'optimal_k': optimal_k,
            'k_values': list(k_range),
            'scores': scores,
            'best_score': max(scores)
        }
    
    def plot_k_optimization(self, k_values, scores):
        """
        绘制K值优化图
        
        Parameters:
        -----------
        k_values : list
            K值列表
        scores : list
            对应的分数列表
        """
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, scores, 'bo-')
        plt.xlabel('Number of Neighbors (K)')
        plt.ylabel('Cross-Validation Score')
        plt.title('KNN: K Value Optimization')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


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
    model = KNNModel(task_type='classification', n_neighbors=5)
    model.fit(X_train, y_train)
    
    # 评估
    metrics = model.evaluate(X_test, y_test)
    print("KNN Classification Model Performance:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics['Classification Report'])
    
    # 寻找最优K值
    print("\nFinding optimal K...")
    results = model.find_optimal_k(X_train, y_train, k_range=range(1, 21), cv=5)
    print(f"Optimal K: {results['optimal_k']}")
    print(f"Best Score: {results['best_score']:.4f}")
    
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
    model = KNNModel(task_type='regression', n_neighbors=5)
    model.fit(X_train, y_train)
    
    # 评估
    metrics = model.evaluate(X_test, y_test)
    print("KNN Regression Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return model


if __name__ == "__main__":
    print("=" * 50)
    print("KNN Classification Example")
    print("=" * 50)
    clf_model = example_usage_classification()
    
    print("\n" + "=" * 50)
    print("KNN Regression Example")
    print("=" * 50)
    reg_model = example_usage_regression()
