"""
Ensemble Learning Methods Template
用于美赛C/E/F题的集成学习算法模板
包括 Gradient Boosting, AdaBoost, XGBoost, LightGBM, Voting, Stacking
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier,
                              AdaBoostRegressor, AdaBoostClassifier,
                              VotingRegressor, VotingClassifier,
                              StackingRegressor, StackingClassifier)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt


class GradientBoostingModel:
    """
    梯度提升模型类
    支持回归和分类任务
    """
    
    def __init__(self, task_type='regression', n_estimators=100, 
                 learning_rate=0.1, max_depth=3, random_state=42):
        """
        初始化模型
        
        Parameters:
        -----------
        task_type : str, default='regression'
            任务类型: 'regression' 或 'classification'
        n_estimators : int, default=100
            提升阶段的数量
        learning_rate : float, default=0.1
            学习率
        max_depth : int, default=3
            单个树的最大深度
        random_state : int, default=42
            随机种子
        """
        self.task_type = task_type
        self.feature_names = None
        
        if task_type == 'regression':
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
    
    def fit(self, X, y):
        """训练模型"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        
        if self.task_type == 'regression':
            return {
                'MSE': mean_squared_error(y, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                'MAE': mean_absolute_error(y, y_pred),
                'R2': r2_score(y, y_pred)
            }
        else:
            return {
                'Accuracy': accuracy_score(y, y_pred),
                'Classification Report': classification_report(y, y_pred)
            }
    
    def get_feature_importance(self):
        """获取特征重要性"""
        importances = self.model.feature_importances_
        
        if self.feature_names:
            return pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        else:
            return importances


class AdaBoostModel:
    """
    AdaBoost模型类
    支持回归和分类任务
    """
    
    def __init__(self, task_type='regression', n_estimators=50, 
                 learning_rate=1.0, random_state=42):
        """
        初始化模型
        
        Parameters:
        -----------
        task_type : str, default='regression'
            任务类型: 'regression' 或 'classification'
        n_estimators : int, default=50
            弱学习器数量
        learning_rate : float, default=1.0
            学习率
        random_state : int, default=42
            随机种子
        """
        self.task_type = task_type
        self.feature_names = None
        
        if task_type == 'regression':
            self.model = AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state
            )
        else:
            self.model = AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state
            )
    
    def fit(self, X, y):
        """训练模型"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        
        if self.task_type == 'regression':
            return {
                'MSE': mean_squared_error(y, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                'MAE': mean_absolute_error(y, y_pred),
                'R2': r2_score(y, y_pred)
            }
        else:
            return {
                'Accuracy': accuracy_score(y, y_pred),
                'Classification Report': classification_report(y, y_pred)
            }
    
    def get_feature_importance(self):
        """获取特征重要性"""
        importances = self.model.feature_importances_
        
        if self.feature_names:
            return pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        else:
            return importances


class XGBoostModel:
    """
    XGBoost模型类
    需要安装xgboost库: pip install xgboost
    """
    
    def __init__(self, task_type='regression', n_estimators=100, 
                 learning_rate=0.1, max_depth=3, random_state=42):
        """
        初始化模型
        
        Parameters:
        -----------
        task_type : str, default='regression'
            任务类型: 'regression' 或 'classification'
        n_estimators : int, default=100
            树的数量
        learning_rate : float, default=0.1
            学习率
        max_depth : int, default=3
            树的最大深度
        random_state : int, default=42
            随机种子
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("Please install xgboost: pip install xgboost")
        
        self.task_type = task_type
        self.feature_names = None
        
        if task_type == 'regression':
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
    
    def fit(self, X, y):
        """训练模型"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        
        if self.task_type == 'regression':
            return {
                'MSE': mean_squared_error(y, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                'MAE': mean_absolute_error(y, y_pred),
                'R2': r2_score(y, y_pred)
            }
        else:
            return {
                'Accuracy': accuracy_score(y, y_pred),
                'Classification Report': classification_report(y, y_pred)
            }
    
    def get_feature_importance(self):
        """获取特征重要性"""
        importances = self.model.feature_importances_
        
        if self.feature_names:
            return pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        else:
            return importances


class VotingEnsemble:
    """
    投票集成模型
    结合多个模型的预测结果
    """
    
    def __init__(self, estimators, task_type='regression', voting='soft'):
        """
        初始化模型
        
        Parameters:
        -----------
        estimators : list of (str, estimator) tuples
            基学习器列表
        task_type : str, default='regression'
            任务类型: 'regression' 或 'classification'
        voting : str, default='soft'
            投票方式 (仅分类): 'hard' 或 'soft'
        """
        self.task_type = task_type
        
        if task_type == 'regression':
            self.model = VotingRegressor(estimators=estimators)
        else:
            self.model = VotingClassifier(estimators=estimators, voting=voting)
    
    def fit(self, X, y):
        """训练模型"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        
        if self.task_type == 'regression':
            return {
                'MSE': mean_squared_error(y, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                'MAE': mean_absolute_error(y, y_pred),
                'R2': r2_score(y, y_pred)
            }
        else:
            return {
                'Accuracy': accuracy_score(y, y_pred),
                'Classification Report': classification_report(y, y_pred)
            }


class StackingEnsemble:
    """
    堆叠集成模型
    使用元学习器组合多个基学习器
    """
    
    def __init__(self, estimators, final_estimator, task_type='regression'):
        """
        初始化模型
        
        Parameters:
        -----------
        estimators : list of (str, estimator) tuples
            基学习器列表
        final_estimator : estimator
            元学习器
        task_type : str, default='regression'
            任务类型: 'regression' 或 'classification'
        """
        self.task_type = task_type
        
        if task_type == 'regression':
            self.model = StackingRegressor(
                estimators=estimators,
                final_estimator=final_estimator
            )
        else:
            self.model = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator
            )
    
    def fit(self, X, y):
        """训练模型"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        
        if self.task_type == 'regression':
            return {
                'MSE': mean_squared_error(y, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
                'MAE': mean_absolute_error(y, y_pred),
                'R2': r2_score(y, y_pred)
            }
        else:
            return {
                'Accuracy': accuracy_score(y, y_pred),
                'Classification Report': classification_report(y, y_pred)
            }


# 使用示例
def example_usage():
    """
    集成学习使用示例
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
    
    # 1. Gradient Boosting
    print("=" * 50)
    print("Gradient Boosting Model")
    print("=" * 50)
    gb_model = GradientBoostingModel(task_type='regression', n_estimators=100)
    gb_model.fit(X_train, y_train)
    gb_metrics = gb_model.evaluate(X_test, y_test)
    print("Performance:")
    for metric, value in gb_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 2. AdaBoost
    print("\n" + "=" * 50)
    print("AdaBoost Model")
    print("=" * 50)
    ada_model = AdaBoostModel(task_type='regression', n_estimators=50)
    ada_model.fit(X_train, y_train)
    ada_metrics = ada_model.evaluate(X_test, y_test)
    print("Performance:")
    for metric, value in ada_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 3. Voting Ensemble
    print("\n" + "=" * 50)
    print("Voting Ensemble Model")
    print("=" * 50)
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    
    estimators = [
        ('lr', LinearRegression()),
        ('rf', RandomForestRegressor(n_estimators=10, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=10, random_state=42))
    ]
    
    voting_model = VotingEnsemble(estimators, task_type='regression')
    voting_model.fit(X_train, y_train)
    voting_metrics = voting_model.evaluate(X_test, y_test)
    print("Performance:")
    for metric, value in voting_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return gb_model, ada_model, voting_model


if __name__ == "__main__":
    gb_model, ada_model, voting_model = example_usage()
