"""
更新后的CNN股票预测系统主程序
严格按照Jiang, Kelly & Xiu (2023)论文要求实现
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from data_processing.processor import StockDataProcessor
from data_processing.image_generator import OHLCImageGenerator
from models.cnn_model import StockCNN
from models.trainer import ModelTrainer
from utils.evaluation import ModelEvaluator
from utils.benchmarks import BenchmarkSignals
from utils.config import IMAGE_CONFIG, TRAIN_CONFIG, EVAL_CONFIG, IMAGE_SAVE_PATH

def run_paper_experiment(symbol="SPX", window_days=20, test_mode=True):
    """
    运行论文实验（严格按照论文要求）
    
    Args:
        symbol: 股票代码
        window_days: 时间窗口天数（5/20/60）
        test_mode: 是否为测试模式
    """
    print(f"\n{'='*60}")
    print(f"Jiang, Kelly & Xiu (2023) 论文复现实验")
    print(f"模型: I{window_days}/R{window_days} ({'周' if window_days==5 else '月' if window_days==20 else '季'}策略)")
    print(f"{'='*60}")
    
    if test_mode:
        print("🧪 测试模式：使用10%数据量")
    
    # 1. 数据加载与处理（论文要求：CRSP个股数据，1993-2019）
    print("\n1. 数据加载与预处理...")
    data_fraction = 0.1 if test_mode else 1.0
    processor = StockDataProcessor(data_fraction=data_fraction)
    processor.load_data()
    
    # 获取序列数据（监督期=持有期）
    sequences, labels, dates = processor.get_processed_data(symbol, window_days, window_days)
    print(f"   共生成 {len(sequences)} 个样本")
    print(f"   监督期=持有期: {window_days}天")
    
    # 2. 生成OHLC图像（论文要求：严格黑白，每天3像素）
    print("\n2. 生成OHLC图像...")
    image_generator = OHLCImageGenerator(window_days)
    
    # 按论文要求：训练期1993-2000，测试期2001-2019
    import pandas as pd
    years = pd.to_datetime(dates).year
    train_mask = (years >= 1993) & (years <= 2000)
    test_mask = years >= 2001
    
    print(f"   数据年份范围: {years.min()}-{years.max()}")
    print(f"   训练期样本: {np.sum(train_mask)}")
    print(f"   测试期样本: {np.sum(test_mask)}")
    
    # 基于训练集拟合归一化统计量
    train_sequences = [seq for seq, is_train in zip(sequences, train_mask) if is_train]
    if len(train_sequences) > 0:
        print("   正在计算训练集归一化统计量...")
        image_generator.fit_normalizer(train_sequences)
    
    images = image_generator.generate_batch(sequences)
    print(f"   图像尺寸: {images.shape}")
    
    # 3. 构建CNN模型（论文要求：I5(2块)、I20(3块)、I60(4块)）
    print("\n3. 构建CNN模型...")
    cnn = StockCNN(window_days)
    model = cnn.build_model(input_shape=images.shape[1:])
    cnn.compile_model(learning_rate=TRAIN_CONFIG["learning_rate"])
    
    print("   模型结构:")
    cnn.summary()
    
    # 4. 训练模型（论文要求：5次独立训练取平均）
    print("\n4. 训练模型...")
    trainer = ModelTrainer(model, window_days)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(images, labels, dates)
    
    print(f"   训练集: {X_train.shape[0]} 样本")
    print(f"   验证集: {X_val.shape[0]} 样本") 
    print(f"   测试集: {X_test.shape[0]} 样本")
    
    # 论文要求：每个模型配置独立训练5次取平均
    if not test_mode:
        print(f"   开始{TRAIN_CONFIG['n_ensemble']}次独立训练...")
        models, histories = trainer.train_multiple_runs(
            X_train, X_val, y_train, y_val, 
            n_runs=TRAIN_CONFIG["n_ensemble"]
        )
        
        # 集成预测
        ensemble_pred = trainer.predict_ensemble(models, X_test)
    else:
        # 测试模式：单次训练
        print("   测试模式：单次训练")
        history = trainer.train(X_train, X_val, y_train, y_val)
        ensemble_pred = model.predict(X_test, verbose=0)
    
    # 5. 模型评估（论文要求：等权/价值加权、H-L策略、夏普比率）
    print("\n5. 评估模型性能...")
    evaluator = ModelEvaluator()
    
    # 计算测试集收益率
    test_returns = []
    for i in range(len(y_test)):
        seq_idx = len(sequences) - len(y_test) + i
        if seq_idx + 1 < len(sequences):
            current_price = sequences[seq_idx].iloc[-1]['Adj_Close_calc']
            future_price = sequences[seq_idx + 1].iloc[-1]['Adj_Close_calc']
            test_returns.append((future_price - current_price) / current_price)
        else:
            test_returns.append(0.0)
    
    # 评估预测结果
    results = evaluator.evaluate_predictions(
        predictions=ensemble_pred,
        actual_returns=test_returns,
        holding_period_days=window_days
    )
    
    # 打印评估报告
    evaluator.print_evaluation_report(results, window_days)
    
    # 6. 基准信号对比（论文要求：MOM、STR、WSTR、TREND）
    print("\n6. 基准信号对比...")
    benchmark_calculator = BenchmarkSignals()
    
    # 准备基准信号数据
    stock_data = {symbol: {
        'prices': [seq['Adj_Close_calc'].iloc[-1] for seq in sequences],
        'returns': [seq['Return'].iloc[-1] if not pd.isna(seq['Return'].iloc[-1]) else 0.0 for seq in sequences]
    }}
    
    print("   基准信号表现:")
    for benchmark_type in EVAL_CONFIG["benchmark_signals"]:
        deciles, signals = benchmark_calculator.create_benchmark_portfolios(
            stock_data, benchmark_type
        )
        
        if deciles is not None:
            benchmark_perf = benchmark_calculator.evaluate_benchmark_performance(
                deciles, stock_data, window_days
            )
            
            if benchmark_perf:
                print(f"   {benchmark_type}: H-L收益={benchmark_perf['long_short_return']:.4f}, "
                      f"夏普比率={benchmark_perf['sharpe_ratio']:.4f}")
        else:
            print(f"   {benchmark_type}: 数据不足，跳过")
    
    return results

def main():
    """主函数"""
    print("CNN股票预测系统 - 论文复现版")
    print("Jiang, Kelly & Xiu (2023): (Re-)Imag(in)ing Price Trends")
    print("="*60)
    
    # 论文要求的三个模型：I5/R5, I20/R20, I60/R60
    window_days_list = [5, 20, 60]
    all_results = {}
    
    for window_days in window_days_list:
        try:
            print(f"\n开始 {window_days} 天模型实验...")
            results = run_paper_experiment("SPX", window_days, test_mode=True)
            all_results[window_days] = results
            
            print(f"\n✓ I{window_days}/R{window_days} 模型测试完成")
                
        except Exception as e:
            print(f"\n✗ I{window_days}/R{window_days} 模型测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 打印所有模型的比较结果
    print(f"\n{'='*60}")
    print("论文复现结果总结")
    print(f"{'='*60}")
    print(f"{'模型':^10}{'准确率':^12}{'等权H-L收益':^15}{'等权夏普比率':^15}")
    print("-"*60)
    
    for days, res in all_results.items():
        if res:
            eq_results = res['equal_weight']
            print(f"I{days}/R{days}:^10{eq_results['long_short_return']:^15.4f}{eq_results['long_short_sharpe']:^15.4f}")
    
    print(f"\n{'='*60}")
    print("论文复现完成！")
    print("注意：当前使用指数数据，完整复现需要CRSP个股数据")
    print("如需完整训练，请修改 test_mode=False")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
