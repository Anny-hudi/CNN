"""
OHLC图像生成测试脚本
测试和验证5天、20天、60天的OHLC图像生成是否符合要求
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from data_processing.processor import StockDataProcessor
from data_processing.image_generator import OHLCImageGenerator
from utils.config import IMAGE_CONFIG, IMAGE_SAVE_PATH

def validate_image_format(image, window_days):
    """验证图像格式是否符合要求"""
    expected_config = IMAGE_CONFIG[window_days]
    expected_width = expected_config["width"]
    expected_height = expected_config["height"]
    
    print(f"\n{window_days}天图像格式验证:")
    print(f"  期望尺寸: {expected_width} x {expected_height}")
    print(f"  实际尺寸: {image.shape[1]} x {image.shape[0]}")
    print(f"  每天像素宽度: {expected_width / window_days:.1f} (期望3.0)")
    
    # 验证尺寸
    size_correct = (image.shape[0] == expected_height and 
                   image.shape[1] == expected_width)
    print(f"  尺寸正确: {'✓' if size_correct else '✗'}")
    
    # 验证像素值范围（应该是0-255的灰度图）
    min_val, max_val = image.min(), image.max()
    print(f"  像素值范围: {min_val} - {max_val} (期望0-255)")
    range_correct = (min_val >= 0 and max_val <= 255)
    print(f"  像素范围正确: {'✓' if range_correct else '✗'}")
    
    return size_correct and range_correct

def analyze_image_content(image, df_window, window_days):
    """分析图像内容是否符合OHLC要求"""
    print(f"\n{window_days}天图像内容分析:")
    
    # 计算价格和成交量区域
    price_area_height = int(image.shape[0] * 0.8)  # 80%为价格区域
    volume_area_height = image.shape[0] - price_area_height  # 20%为成交量区域
    
    print(f"  价格区域高度: {price_area_height}像素 (占{price_area_height/image.shape[0]*100:.1f}%)")
    print(f"  成交量区域高度: {volume_area_height}像素 (占{volume_area_height/image.shape[0]*100:.1f}%)")
    
    # 检查每天的OHLC结构（每天3像素宽）
    pixels_per_day = 3
    white_pixels_per_day = []
    
    for day in range(window_days):
        day_start = day * pixels_per_day
        day_end = (day + 1) * pixels_per_day
        day_region = image[:price_area_height, day_start:day_end]
        white_pixel_count = np.sum(day_region == 255)
        white_pixels_per_day.append(white_pixel_count)
    
    avg_white_pixels = np.mean(white_pixels_per_day)
    print(f"  每天平均白色像素数: {avg_white_pixels:.1f}")
    print(f"  OHLC结构检测: {'✓' if avg_white_pixels > 5 else '✗'} (每天应有开高低收+均线像素)")
    
    # 检查成交量区域
    volume_region = image[price_area_height:, :]
    volume_white_pixels = np.sum(volume_region == 255)
    print(f"  成交量区域白色像素: {volume_white_pixels}")
    print(f"  成交量条检测: {'✓' if volume_white_pixels > 0 else '✗'}")
    
    # 分析移动平均线（寻找连续的白色像素点）
    ma_pixels = 0
    for row in range(price_area_height):
        for col in range(image.shape[1] - 1):
            if image[row, col] == 255 and image[row, col + 1] == 255:
                ma_pixels += 1
    
    print(f"  可能的移动平均线像素: {ma_pixels}")
    print(f"  移动平均线检测: {'✓' if ma_pixels > window_days else '✗'}")

def create_detailed_visualization(image, df_window, window_days, save_path):
    """创建详细的图像可视化"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 主图像显示
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.imshow(image, cmap='gray', aspect='auto', interpolation='nearest')
    ax_main.set_title(f'{window_days}天OHLC图像 (尺寸: {image.shape[1]}×{image.shape[0]})', fontsize=14, fontweight='bold')
    ax_main.set_xlabel('时间（像素）')
    ax_main.set_ylabel('价格水平（像素）')
    
    # 添加网格线显示每天的分界
    for day in range(1, window_days):
        x = day * 3
        ax_main.axvline(x=x-0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # 价格区域和成交量区域分界线
    price_volume_boundary = int(image.shape[0] * 0.8)
    ax_main.axhline(y=price_volume_boundary-0.5, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='价格/成交量分界')
    ax_main.legend()
    
    # 原始价格数据图表
    ax_price = fig.add_subplot(gs[1, 0])
    dates_range = range(len(df_window))
    ax_price.plot(dates_range, df_window['Adj_Open'], 'g-', label='开盘', marker='o', markersize=3)
    ax_price.plot(dates_range, df_window['Adj_High'], 'r-', label='最高', marker='^', markersize=3)
    ax_price.plot(dates_range, df_window['Adj_Low'], 'b-', label='最低', marker='v', markersize=3)
    ax_price.plot(dates_range, df_window['Adj_Close_calc'], 'k-', label='收盘', marker='s', markersize=3)
    
    # 移动平均线
    ma_values = df_window['Adj_Close_calc'].rolling(window=len(df_window)).mean()
    ax_price.plot(dates_range, ma_values, 'purple', label=f'{window_days}日均线', linewidth=2)
    
    ax_price.set_title('原始价格数据')
    ax_price.set_xlabel('天数')
    ax_price.set_ylabel('价格')
    ax_price.legend(fontsize=8)
    ax_price.grid(True, alpha=0.3)
    
    # 成交量数据图表
    ax_volume = fig.add_subplot(gs[1, 1])
    ax_volume.bar(dates_range, df_window['Volume'], alpha=0.7, color='orange')
    ax_volume.set_title('成交量数据')
    ax_volume.set_xlabel('天数')
    ax_volume.set_ylabel('成交量')
    ax_volume.grid(True, alpha=0.3)
    
    # 图像像素强度分析
    ax_intensity = fig.add_subplot(gs[1, 2])
    
    # 按列计算白色像素数量（显示每天的活跃度）
    daily_intensity = []
    for day in range(window_days):
        day_start = day * 3
        day_end = (day + 1) * 3
        day_white_pixels = np.sum(image[:, day_start:day_end] == 255)
        daily_intensity.append(day_white_pixels)
    
    ax_intensity.bar(range(window_days), daily_intensity, alpha=0.7, color='lightblue')
    ax_intensity.set_title('每日图像白色像素数')
    ax_intensity.set_xlabel('天数')
    ax_intensity.set_ylabel('白色像素数')
    ax_intensity.grid(True, alpha=0.3)
    
    # 价格区域分析
    ax_price_region = fig.add_subplot(gs[2, 0])
    price_region = image[:int(image.shape[0] * 0.8), :]
    ax_price_region.imshow(price_region, cmap='gray', aspect='auto', interpolation='nearest')
    ax_price_region.set_title('价格区域 (上80%)')
    ax_price_region.set_xlabel('时间（像素）')
    ax_price_region.set_ylabel('价格水平（像素）')
    
    # 成交量区域分析
    ax_volume_region = fig.add_subplot(gs[2, 1])
    volume_region = image[int(image.shape[0] * 0.8):, :]
    ax_volume_region.imshow(volume_region, cmap='gray', aspect='auto', interpolation='nearest')
    ax_volume_region.set_title('成交量区域 (下20%)')
    ax_volume_region.set_xlabel('时间（像素）')
    ax_volume_region.set_ylabel('成交量水平（像素）')
    
    # 像素值分布
    ax_hist = fig.add_subplot(gs[2, 2])
    ax_hist.hist(image.flatten(), bins=50, alpha=0.7, color='gray', edgecolor='black')
    ax_hist.set_title('像素值分布')
    ax_hist.set_xlabel('像素值')
    ax_hist.set_ylabel('频次')
    ax_hist.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def test_ohlc_image_generation():
    """测试OHLC图像生成"""
    print("=" * 60)
    print("OHLC图像生成测试")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = IMAGE_SAVE_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化数据处理器
    print("1. 加载股票数据...")
    processor = StockDataProcessor()
    processor.load_data()
    
    # 测试不同时间窗口
    window_days_list = [5, 20, 60]
    symbol = "SPX"  # 使用S&P 500数据
    
    for window_days in window_days_list:
        print(f"\n{'='*40}")
        print(f"测试 {window_days} 天图像生成")
        print(f"{'='*40}")
        
        try:
            # 获取数据序列
            sequences, labels = processor.get_processed_data(symbol, window_days, window_days)
            print(f"生成了 {len(sequences)} 个数据序列")
            
            # 选择一个有代表性的序列（选择中间位置的序列）
            test_sequence = sequences[len(sequences) // 2]
            
            print(f"选择测试序列: 日期范围 {test_sequence.index[0]} 到 {test_sequence.index[-1]}")
            print(f"价格范围: {test_sequence['Adj_Close_calc'].min():.2f} - {test_sequence['Adj_Close_calc'].max():.2f}")
            print(f"成交量范围: {test_sequence['Volume'].min():,} - {test_sequence['Volume'].max():,}")
            
            # 生成OHLC图像
            image_generator = OHLCImageGenerator(window_days)
            image = image_generator.generate_image(test_sequence)
            
            print(f"生成图像尺寸: {image.shape}")
            
            # 验证图像格式
            format_valid = validate_image_format(image, window_days)
            
            # 分析图像内容
            analyze_image_content(image, test_sequence, window_days)
            
            # 创建详细可视化
            save_path = f"{output_dir}/{window_days}days_ohlc_analysis.png"
            create_detailed_visualization(image, test_sequence, window_days, save_path)
            
            # 保存原始图像
            raw_image_path = f"{output_dir}/{window_days}days_raw_image.png"
            plt.figure(figsize=(12, 4))
            plt.imshow(image, cmap='gray', aspect='auto', interpolation='nearest')
            plt.title(f'{window_days}天OHLC原始图像 ({image.shape[1]}×{image.shape[0]}像素)')
            plt.xlabel('时间（像素）')
            plt.ylabel('价格+成交量（像素）')
            plt.colorbar(label='像素值 (0=黑, 255=白)')
            
            # 添加分界线
            for day in range(1, window_days):
                x = day * 3
                plt.axvline(x=x-0.5, color='red', linestyle='--', alpha=0.5)
            
            price_volume_boundary = int(image.shape[0] * 0.8)
            plt.axhline(y=price_volume_boundary-0.5, color='blue', linestyle='--', alpha=0.7, linewidth=2)
            
            plt.tight_layout()
            plt.savefig(raw_image_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✓ {window_days}天图像测试完成")
            print(f"  详细分析保存至: {save_path}")
            print(f"  原始图像保存至: {raw_image_path}")
            
        except Exception as e:
            print(f"✗ {window_days}天图像测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("测试完成！请查看生成的图像以验证OHLC格式是否正确。")
    print(f"所有测试图像保存在: {output_dir}/ 目录中")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_ohlc_image_generation() 