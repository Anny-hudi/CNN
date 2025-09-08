# CNN股票价格预测系统

基于卷积神经网络的股票价格趋势预测系统，使用OHLC图像进行深度学习预测。

## 项目结构

```
CNN_2/
├── data/                          # 数据文件夹
│   ├── *.csv                      # 股票历史数据
├── data_processing/               # 数据处理模块
│   ├── processor.py               # 数据加载与预处理
│   └── image_generator.py         # OHLC图像生成
├── models/                        # 模型模块
│   ├── cnn_model.py              # CNN模型定义
│   └── trainer.py                # 训练逻辑
├── utils/                         # 工具模块
│   ├── config.py                 # 配置参数
│   └── evaluation.py             # 评估指标
├── main.py                       # 主程序入口
└── requirements.txt              # 依赖包
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行系统：
```bash
python main.py
```

## 功能特点

- **数据处理**：自动加载和清洗股票数据，按CRSP方法调整价格
- **图像生成**：生成5天/20天/60天的OHLC蜡烛图，包含移动平均线和成交量
- **CNN模型**：针对不同时间窗口优化的卷积神经网络架构
- **策略评估**：计算十分位组合的夏普比率和长短策略表现
- **可视化**：自动生成训练曲线和示例图像

## 输出结果

系统将生成：
- `sample_images/`：OHLC图像示例
- `results/`：训练历史和评估结果
- 控制台输出：详细的训练和评估报告 