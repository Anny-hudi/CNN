"""
模型训练模块
"""
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import models as keras_models
from utils.config import TRAIN_CONFIG
import pandas as pd

class ModelTrainer:
    def __init__(self, model, window_days):
        self.model = model
        self.window_days = window_days
        self.history = None
        
    def prepare_data(self, images, labels, dates):
        """准备训练数据
        
        Args:
            images: 图像数据
            labels: 标签数据
            dates: 对应的日期序列
        """
        # 将日期转换为年份
        years = pd.to_datetime(dates).year
        
        # 按照文献要求划分数据集
        train_mask = (years >= 1993) & (years <= 2000)  # 1993-2000年用于训练和验证
        test_mask = years >= 2001  # 2001年及以后用于测试
        
        # 划分训练/测试集
        X_train = images[train_mask]
        y_train = np.array(labels)[train_mask]  # 转换为numpy数组
        X_test = images[test_mask]
        y_test = np.array(labels)[test_mask]    # 转换为numpy数组
        
        # 检查正负样本比例
        train_pos_ratio = np.mean(y_train == 1)
        print(f"训练集正样本比例: {train_pos_ratio:.2%}")
        
        # 如果训练集正负样本比例严重不平衡（超过55-45），进行下采样平衡
        if abs(train_pos_ratio - 0.5) > 0.05:
            print("平衡训练集正负样本比例...")
            pos_indices = np.where(y_train == 1)[0]
            neg_indices = np.where(y_train == 0)[0]
            
            # 取较少的那一类的样本数
            min_samples = min(len(pos_indices), len(neg_indices))
            
            # 随机下采样
            if len(pos_indices) > min_samples:
                pos_indices = np.random.choice(pos_indices, min_samples, replace=False)
            if len(neg_indices) > min_samples:
                neg_indices = np.random.choice(neg_indices, min_samples, replace=False)
            
            # 合并indices并排序，保持时间顺序
            balanced_indices = np.sort(np.concatenate([pos_indices, neg_indices]))
            X_train = X_train[balanced_indices]
            y_train = y_train[balanced_indices]
            
            print(f"平衡后训练集正样本比例: {np.mean(y_train == 1):.2%}")
        
        # 将训练集进一步分割为训练和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
        )
        
        # 打印数据集大小
        print(f"\n数据集划分:")
        print(f"训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_val)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 确保标签是浮点数类型
        y_train = y_train.astype('float32')
        y_val = y_val.astype('float32')
        y_test = y_test.astype('float32')
        
        # 确保图像数据是浮点数类型
        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')
        X_test = X_test.astype('float32')
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train(self, X_train, X_val, y_train, y_val):
        """训练模型"""
        # 创建数据集对象
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)
        ).batch(128)
        
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val)
        ).batch(128)
        
        # 早停回调：验证集损失连续2个周期未改善时停止
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,  # 2个周期未改善则停止
            restore_best_weights=True,  # 恢复最佳权重
            verbose=1
        )
        
        # 学习率调度：当验证集损失停滞时降低学习率
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # 每次降低一半
            patience=1,  # 1个周期未改善就降低学习率
            min_lr=1e-18,  # 最小学习率
            verbose=1
        )
        
        # 训练模型
        self.history = self.model.fit(
            train_dataset,
            epochs=50,  # 最大训练轮数
            validation_data=val_dataset,
            callbacks=[early_stop, lr_scheduler],
            verbose=1
        )
        
        return self.history
    
    def train_multiple_runs(self, X_train, X_val, y_train, y_val, n_runs=5):
        """多次训练取平均（降低随机性影响）
        
        Args:
            X_train: 训练集图像
            X_val: 验证集图像
            y_train: 训练集标签
            y_val: 验证集标签
            n_runs: 独立训练次数，默认5次
        """
        trained_models = []
        histories = []
        
        for run in range(n_runs):
            print(f"\n=== 第 {run+1}/{n_runs} 次独立训练 ===")
            
            # 重新初始化模型
            model_copy = keras_models.clone_model(self.model)
            model_copy.compile(
                optimizer=self.model.optimizer,
                loss=self.model.loss,
                metrics=self.model.metrics
            )
            
            # 训练
            trainer = ModelTrainer(model_copy, self.window_days)
            history = trainer.train(X_train, X_val, y_train, y_val)
            
            trained_models.append(model_copy)
            histories.append(history)
            
            # 打印当前训练结果
            val_loss = history.history['val_loss'][-1]
            val_acc = history.history['val_binary_accuracy'][-1]
            val_auc = history.history['val_auc'][-1]
            print(f"\n第 {run+1} 次训练结果:")
            print(f"验证集损失: {val_loss:.4f}")
            print(f"验证集准确率: {val_acc:.2%}")
            print(f"验证集AUC: {val_auc:.4f}")
        
        return trained_models, histories
    
    def predict_ensemble(self, models, X_test):
        """集成预测（平均多次训练的预测结果）
        
        Args:
            models: 训练得到的多个模型
            X_test: 测试集图像
            
        Returns:
            ensemble_pred: 集成预测的概率值（范围0-1）
        """
        predictions = []
        
        # 创建测试数据集
        test_dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(128)
        
        for i, model in enumerate(models, 1):
            print(f"\n计算第 {i} 个模型的预测结果...")
            pred = model.predict(test_dataset, verbose=0)
            predictions.append(pred)
        
        # 平均预测概率
        ensemble_pred = np.mean(predictions, axis=0)
        print("\n已完成集成预测")
        
        return ensemble_pred
    
    def get_training_history(self):
        """获取训练历史"""
        return self.history 