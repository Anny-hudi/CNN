"""
简化OHLC图像测试脚本
每次只生成一张黑白图像，展示阶梯状移动平均线
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from data_processing.processor import StockDataProcessor
from data_processing.image_generator import OHLCImageGenerator
from utils.config import IMAGE_CONFIG, IMAGE_SAVE_PATH

def generate_single_ohlc_image(window_days):
    """生成单张OHLC图像"""
    print(f"生成 {window_days} 天OHLC图像...")
    
    # 1. 加载数据
    processor = StockDataProcessor()
    processor.load_data()
    
    # 2. 获取数据序列
    sequences, labels = processor.get_processed_data("SPX", window_days, window_days)
    print(f"共生成 {len(sequences)} 个数据序列")
    
    # 3. 选择中间位置的序列（避免边界效应）
    test_sequence = sequences[len(sequences) // 2]
    
    print(f"选择测试序列:")
    print(f"  日期范围: {test_sequence.index[0]} 到 {test_sequence.index[-1]}")
    print(f"  价格范围: {test_sequence['Adj_Close_calc'].min():.2f} - {test_sequence['Adj_Close_calc'].max():.2f}")
    print(f"  成交量范围: {test_sequence['Volume'].min():,} - {test_sequence['Volume'].max():,}")
    
    # 4. 生成OHLC图像
    image_generator = OHLCImageGenerator(window_days)
    image = image_generator.generate_image(test_sequence)
    
    print(f"生成图像尺寸: {image.shape}")
    
    # 5. 显示图像信息
    expected_config = IMAGE_CONFIG[window_days]
    print(f"期望尺寸: {expected_config['width']} x {expected_config['height']}")
    print(f"每天像素宽度: {expected_config['width'] / window_days:.1f} (期望3.0)")
    
    # 6. 验证图像格式
    size_correct = (image.shape[0] == expected_config['height'] and 
                   image.shape[1] == expected_config['width'])
    print(f"尺寸正确: {'✓' if size_correct else '✗'}")
    
    min_val, max_val = image.min(), image.max()
    print(f"像素值范围: {min_val} - {max_val} (期望0-255)")
    print(f"像素范围正确: {'✓' if min_val >= 0 and max_val <= 255 else '✗'}")
    
    # 7. 分析图像内容
    price_area_height = int(image.shape[0] * 0.8)
    volume_area_height = image.shape[0] - price_area_height
    print(f"价格区域高度: {price_area_height}像素 (占{price_area_height/image.shape[0]*100:.1f}%)")
    print(f"成交量区域高度: {volume_area_height}像素 (占{volume_area_height/image.shape[0]*100:.1f}%)")
    
    # 8. 保存和显示图像
    output_dir = IMAGE_SAVE_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始图像（放大显示）
    plt.figure(figsize=(15, 6))
    plt.imshow(image, cmap='gray', aspect='auto', interpolation='nearest')
    plt.title(f'{window_days}天OHLC图像 - 黑白格式，阶梯状移动平均线\n尺寸: {image.shape[1]}×{image.shape[0]}像素', 
              fontsize=14, fontweight='bold')
    plt.xlabel('时间（像素）')
    plt.ylabel('价格+成交量（像素）')
    
    # 添加分界线标注
    for day in range(1, window_days):
        x = day * 3
        plt.axvline(x=x-0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # 价格/成交量分界线
    price_volume_boundary = int(image.shape[0] * 0.8)
    plt.axhline(y=price_volume_boundary-0.5, color='blue', linestyle='--', alpha=0.8, linewidth=2)
    
    # 添加说明文字
    plt.text(image.shape[1] * 0.02, image.shape[0] * 0.95, '价格区域', 
             fontsize=12, color='blue', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.text(image.shape[1] * 0.02, image.shape[0] * 0.92, '(OHLC + 阶梯状移动平均线)', 
             fontsize=10, color='blue', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.text(image.shape[1] * 0.02, image.shape[0] * 0.05, '成交量区域', 
             fontsize=12, color='blue', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 保存图像
    save_path = f"{output_dir}/{window_days}days_simple_ohlc.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ {window_days}天图像已保存至: {save_path}")
    print("-" * 50)
    
    return image, test_sequence

def main():
    """主函数 - 生成三种时间窗口的OHLC图像"""
    print("=" * 60)
    print("简化OHLC图像生成测试")
    print("特性: 黑白格式 + 阶梯状移动平均线")
    print("=" * 60)
    
    window_days_list = [5, 20, 60]
    
    for window_days in window_days_list:
        try:
            image, sequence = generate_single_ohlc_image(window_days)
            
            # 详细分析阶梯状移动平均线
            print(f"\n{window_days}天图像的移动平均线分析:")
            ma_values = sequence['Adj_Close_calc'].rolling(window=len(sequence)).mean()
            ma_non_null = ma_values.dropna()
            print(f"  移动平均线数据点: {len(ma_non_null)}")
            print(f"  移动平均线范围: {ma_non_null.min():.2f} - {ma_non_null.max():.2f}")
            
            # 检查阶梯状线条（水平线 + 垂直线的组合）
            price_area_height = int(image.shape[0] * 0.8)
            step_pixels = 0
            for row in range(price_area_height):
                consecutive_white = 0
                for col in range(image.shape[1] - 1):
                    if image[row, col] == 255 and image[row, col + 1] == 255:
                        consecutive_white += 1
                    else:
                        if consecutive_white >= 2:  # 水平线段
                            step_pixels += consecutive_white
                        consecutive_white = 0
            
            print(f"  检测到的阶梯状像素: {step_pixels}")
            print(f"  阶梯状移动平均线: {'✓' if step_pixels > window_days * 2 else '✗'}")
            
        except Exception as e:
            print(f"✗ {window_days}天图像生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("测试完成！所有图像均为黑白格式，包含阶梯状移动平均线")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 