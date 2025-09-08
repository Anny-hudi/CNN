"""
测试60天模型构建
"""
import os
import numpy as np
from data_processing.processor import StockDataProcessor
from data_processing.image_generator import OHLCImageGenerator
from models.cnn_model import StockCNN

def test_60day_model():
    """测试60天模型构建"""
    print("测试60天模型构建（优化后）")
    print("="*40)
    
    # 设置实用配置
    os.environ['USE_PRACTICAL_CONFIG'] = '1'
    
    try:
        # 1. 使用更少的数据
        print("1. 加载数据（5%数据量）...")
        processor = StockDataProcessor(data_fraction=0.05)  # 进一步减少到5%
        processor.load_data()
        
        # 2. 获取少量序列
        sequences, labels = processor.get_processed_data("SPX", 60, 60)
        print(f"   生成序列数量: {len(sequences)}")
        
        # 3. 生成少量图像
        print("2. 生成OHLC图像...")
        image_generator = OHLCImageGenerator(60)
        images = image_generator.generate_batch(sequences[:50])  # 只用50个样本
        print(f"   图像数组形状: {images.shape}")
        
        # 4. 构建模型
        print("3. 构建CNN模型...")
        cnn = StockCNN(60)
        model = cnn.build_model(input_shape=images.shape[1:])
        
        print(f"   模型层数: {len(model.layers)}")
        print("   模型结构摘要:")
        model.summary()
        
        print("\n✅ 60天模型构建成功！")
        
    except Exception as e:
        print(f"\n❌ 60天模型构建失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_60day_model() 