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
            # 卷积层
            x = layers.Conv2D(
                filters=block["filters"],
                kernel_size=block["kernel"],  # 5×3滤波器
                padding='valid',  # 无填充
                kernel_initializer='glorot_uniform'  # Xavier初始化
            )(x)
            
            # 添加批归一化层（减少内部协变量偏移）
            x = layers.BatchNormalization()(x)
            
            # 泄漏ReLU激活函数：LeakyReLU(x) = max(0.01x, x)
            x = layers.LeakyReLU(negative_slope=0.01)(x)
            
            # 最大池化层：2×1滤波器（垂直2像素、水平1像素）
            x = layers.MaxPooling2D(pool_size=block["pool"])(x)
        
        # 使用全局平均池化显著降低参数量，避免Flatten导致的超大向量
        x = layers.GlobalAveragePooling2D()(x)
        
        # 根据不同天数设计更小的全连接层神经元数量（实用配置）
        fc_neurons = self._get_fc_neurons(self.window_days)
        x = layers.Dense(
            fc_neurons,
            kernel_initializer='glorot_uniform'  # Xavier初始化
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(negative_slope=0.01)(x)
        x = layers.Dropout(0.5)(x)
        
        # 输出层（单个神经元，sigmoid激活函数）
        outputs = layers.Dense(
            1,
            activation='sigmoid',
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
    
    def compile_model(self, learning_rate=1e-15):
        """编译模型
        
        Args:
            learning_rate: 初始学习率，默认为1e-15
        """
        if self.model is None:
            raise ValueError("请先调用build_model构建模型")
            
        # 使用Adam优化器，初始学习率为1e-15
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
        
        # 使用二元交叉熵损失函数：L(y, ŷ) = -y log(ŷ) - (1-y)log(1-ŷ)
        self.model.compile(
            optimizer=optimizer,
            loss=losses.BinaryCrossentropy(from_logits=False),
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