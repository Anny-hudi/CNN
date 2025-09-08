"""
测试减少数据量的CNN模型构建
"""
import os
import numpy as np
from data_processing.processor import StockDataProcessor
from data_processing.image_generator import OHLCImageGenerator
from models.cnn_model import StockCNN
from utils.config import IMAGE_CONFIG

def test_reduced_data_cnn():
    """测试使用10%数据量的CNN模型构建"""
    print("=" * 60)
    print("测试10%数据量的CNN模型构建")
    print("=" * 60)
    
    # 设置使用实用配置以节省内存
    os.environ['USE_PRACTICAL_CONFIG'] = '1'
    
    window_days_list = [5, 20, 60]
    
    for window_days in window_days_list:
        print(f"\n{'='*40}")
        print(f"测试 {window_days} 天模型")
        print(f"{'='*40}")
        
        try:
            # 1. 使用10%数据量加载数据
            print("1. 加载数据（10%数据量）...")
            processor = StockDataProcessor(data_fraction=0.1)
            processor.load_data()
            
            # 2. 获取数据序列
            sequences, labels = processor.get_processed_data("SPX", window_days, window_days)
            print(f"   生成序列数量: {len(sequences)}")
            print(f"   标签分布: 涨 {sum(labels)} / 跌 {len(labels) - sum(labels)}")
            
            # 3. 生成OHLC图像
            print("2. 生成OHLC图像...")
            image_generator = OHLCImageGenerator(window_days)
            images = image_generator.generate_batch(sequences[:100])  # 只取前100个样本测试
            print(f"   图像数组形状: {images.shape}")
            print(f"   图像数值范围: {images.min():.3f} - {images.max():.3f}")
            
            # 4. 构建CNN模型
            print("3. 构建CNN模型...")
            cnn = StockCNN(window_days)
            model = cnn.build_model(input_shape=images.shape[1:])
            
            print(f"   模型输入形状: {images.shape[1:]}")
            print(f"   模型层数: {len(model.layers)}")
            
            # 5. 编译模型
            print("4. 编译模型...")
            cnn.compile_model(learning_rate=1e-5)
            
            # 6. 显示模型摘要
            print("5. 模型结构摘要:")
            model.summary()
            
            print(f"✓ {window_days}天模型构建成功！")
            print(f"  内存使用: 使用实用配置")
            print(f"  数据量: 10%原始数据")
            print(f"  样本数: {len(images)}")
            
        except Exception as e:
            print(f"✗ {window_days}天模型构建失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("测试完成！")
    print("修改说明:")
    print("• 数据量: 使用10%原始数据（data_fraction=0.1）")
    print("• 神经元配置: 使用实用配置以节省内存")
    print("• 测试样本: 每个窗口只使用前100个样本")
    print("• 内存优化: 启用USE_PRACTICAL_CONFIG环境变量")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_reduced_data_cnn() 