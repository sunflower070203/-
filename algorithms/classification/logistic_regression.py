"""
Logistic Regression Template
用于美赛C/E/F题的逻辑回归模板
适用于二分类和多分类问题
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns


class LogisticRegressionModel:
    """
    逻辑回归模型类
    适用于分类问题
    """
    
    def __init__(self, penalty='l2', C=1.0, max_iter=1000, 
                 normalize=True, random_state=42):
        """
        初始化模型
        
        Parameters:
        -----------
        penalty : str, default='l2'
            正则化类型: 'l1', 'l2', 'elasticnet', 'none'
        C : float, default=1.0
            正则化强度的倒数
        max_iter : int, default=1000
            最大迭代次数
        normalize : bool, default=True
            是否对特征进行标准化
        random_state : int, default=42
            随机种子
        """
        self.model = LogisticRegression(
            penalty=penalty,
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs' if penalty == 'l2' else 'saga'
        )
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
        预测类别
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            测试特征
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            预测类别
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if self.normalize:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        预测概率
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            测试特征
            
        Returns:
        --------
        proba : array, shape (n_samples, n_classes)
            预测概率
        """
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
            真实标签
            
        Returns:
        --------
        metrics : dict
            评估指标
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, average='weighted'),
            'Recall': recall_score(y, y_pred, average='weighted'),
            'F1-Score': f1_score(y, y_pred, average='weighted'),
            'Classification Report': classification_report(y, y_pred),
            'Confusion Matrix': confusion_matrix(y, y_pred)
        }
        
        # 对于二分类，计算AUC
        if len(np.unique(y)) == 2:
            metrics['ROC-AUC'] = roc_auc_score(y, y_proba[:, 1])
        
        return metrics
    
    def get_coefficients(self):
        """
        获取模型系数
        
        Returns:
        --------
        coef_df : DataFrame
            特征名称和对应系数
        """
        if self.feature_names:
            if len(self.model.coef_) == 1:
                # 二分类
                return pd.DataFrame({
                    'Feature': self.feature_names,
                    'Coefficient': self.model.coef_[0]
                }).sort_values('Coefficient', key=abs, ascending=False)
            else:
                # 多分类
                coef_dict = {'Feature': self.feature_names}
                for i, coef in enumerate(self.model.coef_):
                    coef_dict[f'Class_{i}'] = coef
                return pd.DataFrame(coef_dict)
        else:
            return self.model.coef_
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """
        绘制混淆矩阵
        
        Parameters:
        -----------
        y_true : array-like
            真实标签
        y_pred : array-like
            预测标签
        labels : list, default=None
            类别标签
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(self, X, y):
        """
        绘制ROC曲线（仅二分类）
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            测试特征
        y : array-like, shape (n_samples,)
            真实标签
        """
        if len(np.unique(y)) != 2:
            raise ValueError("ROC curve only for binary classification")
        
        y_proba = self.predict_proba(X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.show()


# 使用示例
def example_usage():
    """
    使用示例
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
    model = LogisticRegressionModel(penalty='l2', C=1.0, normalize=True)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    metrics = model.evaluate(X_test, y_test)
    print("Model Performance:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
    print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
    
    print("\nClassification Report:")
    print(metrics['Classification Report'])
    
    # 获取系数
    print("\nTop 10 Feature Coefficients:")
    print(model.get_coefficients().head(10))
    
    return model, X_test, y_test, y_pred


if __name__ == "__main__":
    model, X_test, y_test, y_pred = example_usage()
