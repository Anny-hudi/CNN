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
            {"filters": 64, "kernel": (5, 3), "pool": (2, 1)},
            {"filters": 128, "kernel": (5, 3), "pool": (2, 1)},
            {"filters": 256, "kernel": (5, 3), "pool": (2, 1)},
            {"filters": 512, "kernel": (5, 3), "pool": (2, 1)}
        ]
    }
}

# 训练参数（严格按照论文要求）
TRAIN_CONFIG = {
    "learning_rate": 1e-5,   # 论文要求：初始学习率1e-5
    "batch_size": 128,       # 论文要求：batch size 128
    "epochs": 50,            # 最大训练轮数
    "patience": 2,           # 论文要求：连续2个epoch无改善则停止
    "dropout": 0.5,          # 论文要求：50% dropout
    "train_ratio": 0.7,      # 论文要求：训练期内随机70%/30%划分
    "val_ratio": 0.3,        # 验证集比例
    "n_ensemble": 5,         # 论文要求：每个模型配置独立训练5次取平均
    "optimizer": "Adam",     # 论文要求：Adam优化器
    "weight_init": "Xavier", # 论文要求：Xavier初始化
    "early_stopping": True   # 论文要求：early stopping
}

# 评估参数（论文要求）
EVAL_CONFIG = {
    "n_portfolios": 10,        # 论文要求：十分位组合
    "n_trials": 5,             # 统计显著性检验次数
    "weighting_methods": ["equal", "value"],  # 论文要求：等权/价值加权
    "benchmark_signals": ["MOM", "STR", "WSTR", "TREND"],  # 论文要求：四种基准
    "holding_periods": {       # 论文要求：持有期与监督期一致
        5: 5,    # I5/R5：周策略
        20: 20,  # I20/R20：月策略  
        60: 60   # I60/R60：季策略
    },
    "time_periods": {          # 论文要求：时间区间
        "train_start": "1993-01-01",
        "train_end": "2000-12-31", 
        "test_start": "2001-01-01",
        "test_end": "2019-12-31"
    }
} 