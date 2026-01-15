"""
Model Evaluation Utilities
用于美赛C/E/F题的模型评估工具
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    模型评估类
    提供各种评估指标和可视化工具
    """
    
    @staticmethod
    def evaluate_regression(y_true, y_pred):
        """
        评估回归模型
        
        Parameters:
        -----------
        y_true : array-like
            真实值
        y_pred : array-like
            预测值
            
        Returns:
        --------
        metrics : dict
            评估指标
        """
        return {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    @staticmethod
    def evaluate_classification(y_true, y_pred, y_proba=None, average='weighted'):
        """
        评估分类模型
        
        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        y_proba : array-like, default=None
            预测概率（用于计算AUC）
        average : str, default='weighted'
            多分类平均方式
            
        Returns:
        --------
        metrics : dict
            评估指标
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        # 对于二分类，计算AUC
        if y_proba is not None and len(np.unique(y_true)) == 2:
            if len(y_proba.shape) > 1:
                metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    @staticmethod
    def cross_validate_model(model, X, y, cv=5, scoring=None, task_type='regression'):
        """
        交叉验证
        
        Parameters:
        -----------
        model : estimator
            模型对象
        X : array-like
            特征
        y : array-like
            目标变量
        cv : int, default=5
            交叉验证折数
        scoring : str, default=None
            评分指标
        task_type : str, default='regression'
            任务类型
            
        Returns:
        --------
        results : dict
            交叉验证结果
        """
        if scoring is None:
            scoring = 'r2' if task_type == 'regression' else 'accuracy'
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
    
    @staticmethod
    def plot_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
                           scoring=None, task_type='regression'):
        """
        绘制学习曲线
        
        Parameters:
        -----------
        model : estimator
            模型对象
        X : array-like
            特征
        y : array-like
            目标变量
        cv : int, default=5
            交叉验证折数
        train_sizes : array-like, default=np.linspace(0.1, 1.0, 10)
            训练集大小比例
        scoring : str, default=None
            评分指标
        task_type : str, default='regression'
            任务类型
        """
        if scoring is None:
            scoring = 'r2' if task_type == 'regression' else 'accuracy'
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, train_sizes=train_sizes,
            scoring=scoring, n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, label='Training Score', marker='o')
        plt.fill_between(train_sizes, train_mean - train_std, 
                        train_mean + train_std, alpha=0.15)
        
        plt.plot(train_sizes, val_mean, label='Validation Score', marker='s')
        plt.fill_between(train_sizes, val_mean - val_std, 
                        val_mean + val_std, alpha=0.15)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_validation_curve(model, X, y, param_name, param_range, cv=5,
                             scoring=None, task_type='regression'):
        """
        绘制验证曲线
        
        Parameters:
        -----------
        model : estimator
            模型对象
        X : array-like
            特征
        y : array-like
            目标变量
        param_name : str
            参数名称
        param_range : array-like
            参数范围
        cv : int, default=5
            交叉验证折数
        scoring : str, default=None
            评分指标
        task_type : str, default='regression'
            任务类型
        """
        if scoring is None:
            scoring = 'r2' if task_type == 'regression' else 'accuracy'
        
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, label='Training Score', marker='o')
        plt.fill_between(param_range, train_mean - train_std,
                        train_mean + train_std, alpha=0.15)
        
        plt.plot(param_range, val_mean, label='Validation Score', marker='s')
        plt.fill_between(param_range, val_mean - val_std,
                        val_mean + val_std, alpha=0.15)
        
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(f'Validation Curve for {param_name}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_residuals(y_true, y_pred):
        """
        绘制残差图（回归）
        
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
        axes[0].grid(True)
        
        # 残差直方图
        axes[1].hist(residuals, bins=30, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False):
        """
        绘制混淆矩阵（分类）
        
        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        labels : list, default=None
            类别标签
        normalize : bool, default=False
            是否归一化
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true, y_proba):
        """
        绘制ROC曲线（二分类）
        
        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_proba : array-like
            预测概率
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compare_models(models, X, y, cv=5, task_type='regression'):
        """
        比较多个模型
        
        Parameters:
        -----------
        models : dict
            模型字典，格式为 {name: model}
        X : array-like
            特征
        y : array-like
            目标变量
        cv : int, default=5
            交叉验证折数
        task_type : str, default='regression'
            任务类型
            
        Returns:
        --------
        comparison : DataFrame
            模型比较结果
        """
        results = []
        
        scoring = 'r2' if task_type == 'regression' else 'accuracy'
        
        for name, model in models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            results.append({
                'Model': name,
                'Mean Score': scores.mean(),
                'Std Score': scores.std(),
                'Min Score': scores.min(),
                'Max Score': scores.max()
            })
        
        comparison = pd.DataFrame(results)
        comparison = comparison.sort_values('Mean Score', ascending=False)
        
        return comparison


# 使用示例
def example_usage():
    """
    使用示例
    """
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # 生成示例数据
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    
    # 评估
    evaluator = ModelEvaluator()
    
    metrics = evaluator.evaluate_regression(y_test, y_pred)
    print("回归模型评估:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 比较多个模型
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    comparison = evaluator.compare_models(models, X_train, y_train, cv=5, task_type='regression')
    print("\n模型比较:")
    print(comparison)
    
    return evaluator


if __name__ == "__main__":
    evaluator = example_usage()
