"""
模型训练与评估脚本
"""
import numpy as np
from data_processing.processor import StockDataProcessor
from data_processing.image_generator import OHLCImageGenerator
from models.cnn_model import StockCNN
from models.trainer import ModelTrainer
from utils.evaluation import ModelEvaluator
from utils.config import TRAIN_CONFIG, IMAGE_CONFIG
import pandas as pd

def train_and_evaluate(symbol="SPX", window_days=5):
    """训练模型并评估结果
    
    Args:
        symbol: 股票代码，默认为SPX（标普500）
        window_days: 时间窗口天数，可选5/20/60天
    """
    print(f"\n=== 开始训练 {symbol} {window_days}天模型 ===")
    
    # 1. 数据准备
    print("\n1. 数据加载与预处理...")
    processor = StockDataProcessor()
    processor.load_data()
    sequences, labels, dates = processor.get_processed_data(symbol, window_days, prediction_days=1)
    
    # 2. 数据集划分
    print("\n2. 划分数据集...")
    # 先划分数据，这样我们知道哪些是训练集
    trainer = ModelTrainer(None, window_days)  # 暂时不需要模型
    dates_series = pd.Series(dates)
    years = pd.to_datetime(dates_series).dt.year
    train_indices = np.where((years >= 1993) & (years <= 2000))[0]
    
    # 3. 图像生成与归一化
    print("\n3. 生成OHLC图像...")
    generator = OHLCImageGenerator(window_days)
    
    # 首先只处理训练集图像以计算统计量
    print("计算训练集图像统计量...")
    train_sequences = [sequences[i] for i in train_indices]
    generator.fit_normalizer(train_sequences)
    
    # 现在生成所有归一化后的图像
    print("生成所有归一化图像...")
    images = generator.generate_batch(sequences)
    
    # 4. 准备训练数据
    print("\n4. 准备训练数据...")
    input_shape = (IMAGE_CONFIG[window_days]["height"], 
                  IMAGE_CONFIG[window_days]["width"], 1)
    
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        images=images,
        labels=np.array(labels),
        dates=dates
    )
    
    # 5. 构建模型
    print("\n5. 构建CNN模型...")
    model = StockCNN(window_days)
    cnn = model.build_model(input_shape)
    model.compile_model(learning_rate=TRAIN_CONFIG["learning_rate"])
    print("\n模型结构:")
    model.summary()
    
    # 更新trainer的模型
    trainer = ModelTrainer(cnn, window_days)
    
    # 6. 训练模型
    print(f"\n6. 开始训练（{TRAIN_CONFIG['n_ensemble']}次独立训练）...")
    models, histories = trainer.train_multiple_runs(
        X_train, X_val, y_train, y_val,
        n_runs=TRAIN_CONFIG["n_ensemble"]
    )
    
    # 7. 评估结果
    print("\n7. 评估模型性能...")
    evaluator = ModelEvaluator()
    
    # 集成预测
    ensemble_pred = trainer.predict_ensemble(models, X_test)
    
    # 计算测试集收益率（用于评估）
    test_returns = []
    for i in range(len(y_test)):
        seq_idx = len(sequences) - len(y_test) + i
        current_price = sequences[seq_idx].iloc[-1]['Adj_Close_calc']
        next_price = sequences[seq_idx+1].iloc[-1]['Adj_Close_calc']
        test_returns.append((next_price - current_price) / current_price)
    
    # 评估预测结果
    results = evaluator.evaluate_predictions(
        predictions=ensemble_pred,
        actual_returns=test_returns
    )
    
    # 打印评估报告
    evaluator.print_evaluation_report(results, window_days)
    
    return models, results

if __name__ == "__main__":
    # 训练所有时间窗口的模型
    window_days_list = [5, 20, 60]
    all_results = {}
    
    for days in window_days_list:
        print(f"\n{'='*50}")
        print(f"训练 {days} 天模型")
        print(f"{'='*50}")
        
        models, results = train_and_evaluate(window_days=days)
        all_results[days] = results
        
    # 打印所有模型的比较结果
    print("\n\n最终结果比较:")
    print("="*50)
    print(f"{'天数':^10}{'准确率':^15}{'多空收益':^15}{'夏普比率':^15}")
    print("-"*50)
    
    for days, res in all_results.items():
        print(f"{days:^10}{res['accuracy']:^15.4f}"
              f"{res['long_short_return']:^15.4f}"
              f"{res['long_short_sharpe']:^15.4f}") 