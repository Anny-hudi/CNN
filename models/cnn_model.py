"""
CNN模型定义
"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics, losses
from utils.config import MODEL_CONFIG

class StockCNN:
    def __init__(self, window_days):
        self.window_days = window_days
        self.config = MODEL_CONFIG[window_days]
        self.model = None
        
    def build_model(self, input_shape):
        """构建CNN模型"""
        # 使用Input层明确指定输入形状
        inputs = layers.Input(shape=input_shape)
        
        x = inputs
        # 按照文献要求构建每个构建块
        for i, block in enumerate(self.config["blocks"]):
            # 卷积层：kernel_size=3, padding=1（论文要求）
            x = layers.Conv2D(
                filters=block["filters"],
                kernel_size=3,  # 论文要求3x3卷积核
                padding='same',  # 论文要求padding=1
                kernel_initializer='glorot_uniform'  # Xavier初始化
            )(x)
            
            # 添加批归一化层（论文要求）
            x = layers.BatchNormalization()(x)
            
            # 泄漏ReLU激活函数（论文要求）
            x = layers.LeakyReLU(negative_slope=0.01)(x)
            
            # 最大池化层：2×2滤波器（论文要求）
            x = layers.MaxPooling2D(pool_size=2)(x)
        
        # 按照论文要求：Flatten → Linear → Dropout(0.5) → ReLU → Linear(2) → Softmax
        x = layers.Flatten()(x)
        
        # 第一个全连接层
        x = layers.Dense(
            512,  # 论文示例通道数
            kernel_initializer='glorot_uniform'  # Xavier初始化
        )(x)
        x = layers.Dropout(0.5)(x)  # 论文要求50% dropout
        x = layers.ReLU()(x)  # 论文要求ReLU激活
        
        # 输出层：2个神经元（上下概率）
        outputs = layers.Dense(
            2,  # 论文要求输出2个类别
            activation='softmax',  # 论文要求Softmax激活
            kernel_initializer='glorot_uniform'  # Xavier初始化
        )(x)
        
        # 创建模型
        self.model = models.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def _get_fc_neurons(self, window_days):
        """根据文献要求获取全连接层神经元数量"""
        # 为避免OOM，统一采用实用的小规模配置
        practical_config = {
            5: 128,    # 进一步减少
            20: 256,   # 进一步减少
            60: 128    # 大幅减少60天模型参数
        }
        return practical_config.get(window_days, 128)
    
    def compile_model(self, learning_rate=1e-5):
        """编译模型
        
        Args:
            learning_rate: 初始学习率，论文要求1e-5
        """
        if self.model is None:
            raise ValueError("请先调用build_model构建模型")
            
        # 使用Adam优化器，初始学习率为1e-5（论文要求）
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        
        # 使用Keras原生评估指标
        metrics_list = [
            'binary_accuracy',  # 二分类准确率
            metrics.AUC(name='auc'),  # ROC曲线下面积
            metrics.Precision(name='precision'),  # 精确率
            metrics.Recall(name='recall'),  # 召回率
            metrics.TruePositives(name='tp'),  # 真正例
            metrics.TrueNegatives(name='tn'),  # 真负例
            metrics.FalsePositives(name='fp'),  # 假正例
            metrics.FalseNegatives(name='fn')   # 假负例
        ]
        
        # 使用交叉熵损失函数（论文要求）
        self.model.compile(
            optimizer=optimizer,
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=metrics_list
        )
    
    def get_model(self):
        """获取模型"""
        return self.model
    
    def summary(self):
        """打印模型结构"""
        if self.model:
            return self.model.summary()
        else:
            print("模型尚未构建") 