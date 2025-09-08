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
        
        # 全连接层
        x = layers.Flatten()(x)
        
        # 根据不同天数设计不同的全连接层神经元数量
        fc_neurons = self._get_fc_neurons(self.window_days)
        x = layers.Dense(
            fc_neurons,
            kernel_initializer='glorot_uniform'  # Xavier初始化
        )(x)
        x = layers.BatchNormalization()(x)  # 全连接层后也添加批归一化
        x = layers.LeakyReLU(negative_slope=0.01)(x)
        x = layers.Dropout(0.5)(x)  # 50% dropout防止过拟合
        
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
        # 文献要求的神经元数量（可能在某些硬件上内存不足）
        literature_config = {
            5: 15360,   # 5天图像：15360神经元
            20: 46080,  # 20天图像：46080神经元
            60: 184320  # 60天图像：184320神经元
        }
        
        # 实用的神经元数量配置（考虑硬件限制）
        practical_config = {
            5: 15360,   # 5天：保持不变（内存可接受）
            20: 8192,   # 20天：减少到8192（约为文献要求的1/6）
            60: 4096    # 60天：进一步减少到4096（约为文献要求的1/45）
        }
        
        # 默认使用文献要求，如果内存不足可以切换到实用配置
        # 可通过环境变量 USE_PRACTICAL_CONFIG=1 来切换
        import os
        use_practical = os.environ.get('USE_PRACTICAL_CONFIG', '0') == '1'
        
        if use_practical:
            print(f"使用实用配置：{window_days}天模型 → {practical_config[window_days]}神经元")
            return practical_config.get(window_days, 128)
        else:
            return literature_config.get(window_days, 128)
    
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