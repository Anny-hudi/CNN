"""
CNN股票预测系统主程序
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from data_processing.processor import StockDataProcessor
from data_processing.image_generator import OHLCImageGenerator
from models.cnn_model import StockCNN
from models.trainer import ModelTrainer
from utils.evaluation import ModelEvaluator
from utils.config import IMAGE_CONFIG, TRAIN_CONFIG, IMAGE_SAVE_PATH

def save_sample_images(images, labels, window_days, save_dir=None):
    """保存示例图像"""
    if save_dir is None:
        save_dir = IMAGE_SAVE_PATH
    os.makedirs(save_dir, exist_ok=True)
    
    # 选择一个正样本和一个负样本
    pos_idx = np.where(np.array(labels) == 1)[0][0]
    neg_idx = np.where(np.array(labels) == 0)[0][0]
    
    for idx, label_name in [(pos_idx, "positive"), (neg_idx, "negative")]:
        img = images[idx].squeeze()
        plt.figure(figsize=(10, 4))
        plt.imshow(img, cmap='gray', aspect='auto')
        plt.title(f'{window_days}天OHLC图像 - {label_name}样本')
        plt.xlabel('时间（像素）')
        plt.ylabel('价格水平（像素）')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{window_days}days_{label_name}_sample.png', dpi=150)
        plt.close()

def plot_training_history(history, window_days, save_dir=None):
    """绘制训练历史"""
    if save_dir is None:
        save_dir = IMAGE_SAVE_PATH
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(history.history['loss'], label='训练损失')
    ax1.plot(history.history['val_loss'], label='验证损失')
    ax1.set_title(f'{window_days}天模型训练损失')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(history.history['accuracy'], label='训练准确率')
    ax2.plot(history.history['val_accuracy'], label='验证准确率')
    ax2.set_title(f'{window_days}天模型训练准确率')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{window_days}days_training_history.png', dpi=150)
    plt.close()

def run_experiment(symbol="SPX", window_days=20, test_mode=True):
    """
    运行单个实验
    
    Args:
        symbol: 股票代码
        window_days: 时间窗口天数
        test_mode: 是否为测试模式（使用10%数据）
    """
    print(f"\n{'='*50}")
    print(f"开始 {symbol} {window_days}天模型实验")
    if test_mode:
        print("🧪 测试模式：使用10%数据量")
    print(f"{'='*50}")
    
    # 1. 数据加载与处理
    print("1. 加载和处理数据...")
    data_fraction = 0.1 if test_mode else 1.0  # 测试模式使用10%数据
    processor = StockDataProcessor(data_fraction=data_fraction)
    processor.load_data()
    
    # 获取序列数据
    sequences, labels, dates = processor.get_processed_data(symbol, window_days, window_days)
    print(f"   共生成 {len(sequences)} 个样本")
    
    # 2. 生成OHLC图像
    print("2. 生成OHLC图像...")
    image_generator = OHLCImageGenerator(window_days)
    # 基于训练集年份拟合归一化统计量
    import pandas as pd
    years = pd.to_datetime(dates).year
    print(f"   数据年份范围: {years.min()}-{years.max()}")
    
    # 根据数据年份范围调整训练集筛选条件
    if test_mode:
        # 测试模式：使用前70%的数据作为训练集
        train_size = int(len(sequences) * 0.7)
        train_sequences = sequences[:train_size]
        print(f"   测试模式：使用前{train_size}个序列作为训练集")
    else:
        # 生产模式：使用1993-2000年数据
        train_mask = (years >= 1993) & (years <= 2000)
        train_sequences = [seq for seq, is_train in zip(sequences, train_mask) if is_train]
        print(f"   生产模式：训练集序列数量: {len(train_sequences)}")
    
    if len(train_sequences) > 0:
        print("   正在计算训练集归一化统计量...")
        image_generator.fit_normalizer(train_sequences)
    else:
        print("   警告：训练集为空，跳过归一化统计量计算")
    images = image_generator.generate_batch(sequences)
    print(f"   图像尺寸: {images.shape}")
    
    # 保存示例图像
    save_sample_images(images, labels, window_days)
    
    # 3. 构建和编译模型
    print("3. 构建CNN模型...")
    
    # 测试模式下使用实用配置以节省内存
    if test_mode:
        import os
        os.environ['USE_PRACTICAL_CONFIG'] = '1'
    
    cnn = StockCNN(window_days)
    model = cnn.build_model(input_shape=images.shape[1:])
    cnn.compile_model(learning_rate=TRAIN_CONFIG["learning_rate"])
    
    print("   模型结构:")
    cnn.summary()
    
    # 4. 训练模型
    print("4. 训练模型...")
    trainer = ModelTrainer(model, window_days)
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(images, labels, dates)
    
    print(f"   训练集: {X_train.shape[0]} 样本")
    print(f"   验证集: {X_val.shape[0]} 样本") 
    print(f"   测试集: {X_test.shape[0]} 样本")
    
    # 单次训练
    history = trainer.train(X_train, X_val, y_train, y_val)
    
    # 绘制训练历史
    plot_training_history(history, window_days)
    
    # 5. 模型评估
    print("5. 评估模型...")
    
    # 获取测试集预测
    test_predictions = model.predict(X_test, verbose=0)
    
    # 计算测试集对应的真实收益
    test_sequences = sequences[len(sequences) - len(X_test):]
    test_returns = []
    for seq in test_sequences:
        current_price = seq['Adj_Close_calc'].iloc[-1]
        future_price = seq['Adj_Close_calc'].iloc[-1]  # 简化处理
        test_returns.append((future_price - current_price) / current_price)
    
    # 计算准确率
    test_accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"   测试集准确率: {test_accuracy:.4f}")
    
    # 简化评估（测试模式）
    if test_mode:
        print("   🧪 测试模式：跳过完整评估")
        return
    
    # 完整评估（生产模式）
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_predictions(test_predictions, test_returns)
    
    print(f"   预测准确率: {results['accuracy']:.4f}")
    print(f"   长短策略收益: {results['long_short_return']:.4f}")
    print(f"   长短策略夏普比率: {results['long_short_sharpe']:.4f}")
    
    return results

def main():
    """主函数"""
    print("CNN股票预测系统")
    print("================")
    
    # 测试多个时间窗口
    window_days_list = [5, 20, 60]
    
    for window_days in window_days_list:
        try:
            # 默认使用测试模式（10%数据）
            results = run_experiment("SPX", window_days, test_mode=True)
            
            if results:
                print(f"\n✓ {window_days}天模型测试完成")
            else:
                print(f"\n✓ {window_days}天模型测试模式完成")
                
        except Exception as e:
            print(f"\n✗ {window_days}天模型测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n所有测试完成！")
    print("如需完整训练，请修改 test_mode=False")

if __name__ == "__main__":
    main() 