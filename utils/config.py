"""
CNN股票预测系统配置参数
"""
import os

# 数据路径
DATA_PATH = "data"

# 图像保存路径
IMAGE_SAVE_PATH = r"C:\Users\Anny\PycharmProjects\CNN\test_images"

# 图像参数
IMAGE_CONFIG = {
    5: {"width": 15, "height": 32, "days": 5},    # 3像素/天 × 5天，文献要求32×15
    20: {"width": 60, "height": 64, "days": 20},   # 3像素/天 × 20天，文献要求64×60
    60: {"width": 180, "height": 96, "days": 60}   # 3像素/天 × 60天，文献要求96×180
}

# 模型参数
MODEL_CONFIG = {
    5: {
        "blocks": [
            {"filters": 64, "kernel": (5, 3), "pool": (2, 1)},
            {"filters": 128, "kernel": (5, 3), "pool": (2, 1)}
        ]
    },
    20: {
        "blocks": [
            {"filters": 64, "kernel": (5, 3), "pool": (2, 1)},
            {"filters": 128, "kernel": (5, 3), "pool": (2, 1)},
            {"filters": 256, "kernel": (5, 3), "pool": (2, 1)}
        ]
    },
    60: {
        "blocks": [
            {"filters": 32, "kernel": (5, 3), "pool": (2, 1)},
            {"filters": 64, "kernel": (5, 3), "pool": (2, 1)},
            {"filters": 128, "kernel": (5, 3), "pool": (2, 1)}
        ]
    }
}

# 训练参数
TRAIN_CONFIG = {
    "learning_rate": 1e-15,  # 按文献要求修改为1e-15
    "batch_size": 128,       # 保持128不变
    "epochs": 50,
    "patience": 2,           # 2个周期未改善则停止
    "dropout": 0.5,         # 50% dropout
    "train_ratio": 0.7,     # 训练集比例
    "val_ratio": 0.3,       # 验证集比例
    "n_ensemble": 5         # 每个配置训练5次取平均
}

# 评估参数
EVAL_CONFIG = {
    "n_portfolios": 10,     # 十分位组合
    "n_trials": 5          # 统计显著性检验次数
} 