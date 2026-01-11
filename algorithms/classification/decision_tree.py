"""
Decision Tree Template
用于美赛C/E/F题的决策树模板
支持回归和分类任务
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)
import matplotlib.pyplot as plt


class DecisionTreeModel:
    """
    决策树模型类
    支持回归和分类任务
    """
    
    def __init__(self, task_type='classification', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=42):
        """
        初始化模型
        
        Parameters:
        -----------
        task_type : str, default='classification'
            任务类型: 'regression' 或 'classification'
        max_depth : int, default=None
            树的最大深度
        min_samples_split : int, default=2
            分裂内部节点所需的最小样本数
        min_samples_leaf : int, default=1
            叶节点所需的最小样本数
        random_state : int, default=42
            随机种子
        """
        self.task_type = task_type
        self.feature_names = None
        
        if task_type == 'classification':
            self.model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
        else:
            self.model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
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
    
    def get_feature_importance(self):
        """
        获取特征重要性
        
        Returns:
        --------
        importance_df : DataFrame
            特征重要性排序
        """
        importances = self.model.feature_importances_
        
        if self.feature_names:
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            })
        else:
            importance_df = pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(len(importances))],
                'Importance': importances
            })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        return importance_df
    
    def plot_tree(self, max_depth=3, figsize=(20, 10)):
        """
        绘制决策树
        
        Parameters:
        -----------
        max_depth : int, default=3
            显示的最大深度
        figsize : tuple, default=(20, 10)
            图形大小
        """
        plt.figure(figsize=figsize)
        plot_tree(self.model, 
                 feature_names=self.feature_names,
                 filled=True,
                 rounded=True,
                 max_depth=max_depth)
        plt.title('Decision Tree Visualization')
        plt.tight_layout()
        plt.show()
    
    def get_tree_depth(self):
        """
        获取树的深度
        
        Returns:
        --------
        depth : int
            树的深度
        """
        return self.model.get_depth()
    
    def get_n_leaves(self):
        """
        获取叶节点数量
        
        Returns:
        --------
        n_leaves : int
            叶节点数量
        """
        return self.model.get_n_leaves()
    
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
        
        scoring = 'r2' if self.task_type == 'regression' else 'accuracy'
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }


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
    model = DecisionTreeModel(task_type='classification', max_depth=5)
    model.fit(X_train, y_train)
    
    # 评估
    metrics = model.evaluate(X_test, y_test)
    print("Decision Tree Classification Model Performance:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"\nTree Depth: {model.get_tree_depth()}")
    print(f"Number of Leaves: {model.get_n_leaves()}")
    
    # 特征重要性
    print("\nTop 10 Feature Importances:")
    print(model.get_feature_importance().head(10))
    
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
    model = DecisionTreeModel(task_type='regression', max_depth=5)
    model.fit(X_train, y_train)
    
    # 评估
    metrics = model.evaluate(X_test, y_test)
    print("Decision Tree Regression Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"\nTree Depth: {model.get_tree_depth()}")
    print(f"Number of Leaves: {model.get_n_leaves()}")
    
    return model


if __name__ == "__main__":
    print("=" * 50)
    print("Decision Tree Classification Example")
    print("=" * 50)
    clf_model = example_usage_classification()
    
    print("\n" + "=" * 50)
    print("Decision Tree Regression Example")
    print("=" * 50)
    reg_model = example_usage_regression()
