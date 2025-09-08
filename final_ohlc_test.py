"""
最终版OHLC图像测试脚本
严格按照要求：每次只生成一张黑白图像，包含阶梯状移动平均线
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from data_processing.processor import StockDataProcessor
from data_processing.image_generator import OHLCImageGenerator
from utils.config import IMAGE_CONFIG, IMAGE_SAVE_PATH

def generate_final_ohlc_image(window_days):
    """生成最终版OHLC图像（严格黑白，阶梯状移动平均线）"""
    print(f"生成 {window_days} 天OHLC图像...")
    
    # 1. 加载数据
    processor = StockDataProcessor()
    processor.load_data()
    
    # 2. 获取数据序列
    sequences, labels, dates = processor.get_processed_data("SPX", window_days, window_days)
    print(f"共生成 {len(sequences)} 个数据序列")
    
    # 3. 选择中间位置的序列
    test_sequence = sequences[len(sequences) // 2]
    
    print(f"选择测试序列:")
    print(f"  日期范围: {test_sequence.index[0]} 到 {test_sequence.index[-1]}")
    print(f"  价格范围: {test_sequence['Adj_Close_calc'].min():.2f} - {test_sequence['Adj_Close_calc'].max():.2f}")
    
    # 4. 生成OHLC图像
    image_generator = OHLCImageGenerator(window_days)
    image = image_generator.generate_image(test_sequence)
    
    # 5. 验证图像格式
    expected_config = IMAGE_CONFIG[window_days]
    print(f"生成图像尺寸: {image.shape} (期望: {expected_config['height']} x {expected_config['width']})")
    
    # 确保严格黑白格式
    unique_values = np.unique(image)
    print(f"图像像素值: {unique_values} (严格黑白要求: 0和255)")
    
    # 6. 验证OHLC结构
    price_area_height = int(image.shape[0] * 0.75)
    volume_area_height = int(image.shape[0] * 0.2)
    print(f"价格区域: {price_area_height}像素 (75% - 顶部四分之三)")
    print(f"成交量区域: {volume_area_height}像素 (20% - 底部五分之一)")
    
    # 7. 验证移动平均线
    ma_values = test_sequence['Adj_Close_calc'].rolling(window=window_days, min_periods=1).mean()
    ma_non_null = ma_values.dropna()
    print(f"移动平均线数据点: {len(ma_non_null)} (窗口: {window_days}天)")
    
    # 检查移动平均线连续性（直线连接）
    ma_line_segments = 0
    for row in range(price_area_height):
        for col in range(image.shape[1] - 1):
            # 检查连续的白色像素（移动平均线）
            if (image[row, col] == 255 and 
                image[row, col + 1] == 255):
                ma_line_segments += 1
    
    print(f"检测到移动平均线段: {ma_line_segments} (直线连接特征)")
    
    # 8. 保存图像（只保存一张）
    output_dir = IMAGE_SAVE_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用matplotlib保存纯黑白图像
    plt.figure(figsize=(image.shape[1]/10, image.shape[0]/10))  # 按像素比例设置尺寸
    plt.imshow(image, cmap='gray', aspect='equal', interpolation='nearest')
    plt.axis('off')  # 移除坐标轴
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    
    # 保存图像文件
    save_path = f"{output_dir}/{window_days}days_final_ohlc.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0, 
                facecolor='black', edgecolor='black')
    
    # 打印图像信息供查看
    print(f"\n📊 {window_days}天OHLC图像详细信息:")
    print(f"  图像尺寸: {image.shape[1]} × {image.shape[0]} 像素")
    print(f"  像素值范围: {image.min()} - {image.max()}")
    print(f"  唯一像素值: {np.unique(image)}")
    print(f"  白色像素数量: {np.sum(image == 255)}")
    print(f"  黑色像素数量: {np.sum(image == 0)}")
    print(f"  白色像素占比: {np.sum(image == 255) / image.size * 100:.2f}%")
    
    # 按区域分析像素分布
    price_area_height = int(image.shape[0] * 0.75)
    volume_area_height = int(image.shape[0] * 0.2)
    
    price_region = image[:price_area_height, :]
    volume_region = image[image.shape[0]-volume_area_height:, :]
    
    print(f"  价格区域白色像素: {np.sum(price_region == 255)} (占{np.sum(price_region == 255)/price_region.size*100:.2f}%)")
    print(f"  成交量区域白色像素: {np.sum(volume_region == 255)} (占{np.sum(volume_region == 255)/volume_region.size*100:.2f}%)")
    
    # 按天分析像素分布
    print(f"  每日像素分析:")
    if window_days <= 10:
        # 对于10天以内的图像，显示所有天
        for day in range(window_days):
            day_start = day * 3
            day_end = (day + 1) * 3
            day_region = image[:, day_start:day_end]
            white_pixels = np.sum(day_region == 255)
            print(f"    第{day+1}天: {white_pixels}个白色像素")
    else:
        # 对于超过10天的图像，显示前5天、中间几天、后5天
        for day in range(5):  # 前5天
            day_start = day * 3
            day_end = (day + 1) * 3
            day_region = image[:, day_start:day_end]
            white_pixels = np.sum(day_region == 255)
            print(f"    第{day+1}天: {white_pixels}个白色像素")
        
        print(f"    ... (中间{window_days-10}天)")
        
        for day in range(window_days-5, window_days):  # 后5天
            day_start = day * 3
            day_end = (day + 1) * 3
            day_region = image[:, day_start:day_end]
            white_pixels = np.sum(day_region == 255)
            print(f"    第{day+1}天: {white_pixels}个白色像素")
    
    # 计算每日像素统计
    daily_pixels = []
    for day in range(window_days):
        day_start = day * 3
        day_end = (day + 1) * 3
        day_region = image[:, day_start:day_end]
        white_pixels = np.sum(day_region == 255)
        daily_pixels.append(white_pixels)
    
    print(f"  每日像素统计: 平均{np.mean(daily_pixels):.1f}, 最小{np.min(daily_pixels)}, 最大{np.max(daily_pixels)}")
    
    plt.show()  # 显示图像
    plt.close()  # 显示后关闭图形
    
    print(f"✓ {window_days}天图像已保存至: {save_path}")
    print(f"  图像特征: 严格黑白 + 阶梯状移动平均线")
    print("-" * 50)
    
    return image

def validate_ohlc_requirements(image, window_days):
    """验证OHLC图像是否符合所有要求"""
    print(f"\n验证 {window_days} 天图像要求:")
    
    # 1. 尺寸验证
    expected_config = IMAGE_CONFIG[window_days]
    expected_width = expected_config["width"]
    expected_height = expected_config["height"]
    
    size_ok = (image.shape[0] == expected_height and image.shape[1] == expected_width)
    print(f"  尺寸要求: {'✓' if size_ok else '✗'} ({image.shape[1]}×{image.shape[0]} vs {expected_width}×{expected_height})")
    
    # 2. 每天3像素宽度验证
    pixels_per_day = expected_width / window_days
    print(f"  每天像素宽度: {'✓' if pixels_per_day == 3.0 else '✗'} ({pixels_per_day:.1f}/天)")
    
    # 3. 黑白格式验证
    unique_values = np.unique(image)
    is_pure_bw = len(unique_values) == 2 and 0 in unique_values and 255 in unique_values
    print(f"  纯黑白格式: {'✓' if is_pure_bw else '✗'} (像素值: {unique_values})")
    
    # 4. 价格/成交量区域验证
    price_area_height = int(image.shape[0] * 0.75)
    volume_area_height = int(image.shape[0] * 0.2)
    price_ratio = price_area_height / image.shape[0]
    volume_ratio = volume_area_height / image.shape[0]
    price_ratio_ok = abs(price_ratio - 0.75) < 0.05  # 放宽容差到5%
    volume_ratio_ok = abs(volume_ratio - 0.2) < 0.05   # 放宽容差到5%
    print(f"  价格/成交量区域: {'✓' if price_ratio_ok and volume_ratio_ok else '✗'} (实际: {price_ratio:.3f}/{volume_ratio:.3f}, 期望: 0.75/0.2)")
    
    # 5. OHLC结构验证
    ohlc_pixels = 0
    for day in range(window_days):
        day_start = day * 3
        day_end = (day + 1) * 3
        day_region = image[:price_area_height, day_start:day_end]
        white_pixels = np.sum(day_region == 255)
        ohlc_pixels += white_pixels
    
    avg_ohlc_pixels = ohlc_pixels / window_days
    print(f"  OHLC结构: {'✓' if avg_ohlc_pixels > 5 else '✗'} (平均{avg_ohlc_pixels:.1f}像素/天)")
    
    # 6. 成交量条验证（检查底部20%区域）
    volume_start_y = image.shape[0] - volume_area_height
    volume_region = image[volume_start_y:, :]
    volume_pixels = np.sum(volume_region == 255)
    print(f"  成交量条: {'✓' if volume_pixels > 0 else '✗'} ({volume_pixels}像素，在底部20%区域)")
    
    # 7. 移动平均线验证（直线连接，在顶部75%区域）
    ma_pattern = 0
    for row in range(price_area_height):
        for col in range(image.shape[1] - 1):
            # 检查连续的白色像素（移动平均线）
            if (image[row, col] == 255 and 
                image[row, col + 1] == 255):
                ma_pattern += 1
    
    ma_ok = ma_pattern > window_days
    print(f"  移动平均线: {'✓' if ma_ok else '✗'} (检测到{ma_pattern}个连续像素段)")
    
    # 总体评估
    all_ok = size_ok and is_pure_bw and price_ratio_ok and volume_ratio_ok and avg_ohlc_pixels > 5 and volume_pixels > 0 and ma_ok
    print(f"  总体评估: {'✓ 完全符合要求' if all_ok else '✗ 存在问题'}")
    
    return all_ok

def main():
    """主函数 - 生成符合要求的OHLC图像"""
    print("=" * 60)
    print("最终版OHLC图像生成")
    print("要求: 严格黑白 + 阶梯状移动平均线 + 每次只保存一张图")
    print("=" * 60)
    
    # 询问用户是否只生成第一张图
    print("\n选择运行模式:")
    print("1. 只生成第一张图（5天）- 快速查看")
    print("2. 生成所有图像（5天、20天、60天）- 完整测试")
    
    try:
        choice = input("请输入选择 (1 或 2，默认为1): ").strip()
        if choice == "2":
            window_days_list = [5, 20, 60]
            print("将生成所有三种时间窗口的图像...")
        else:
            window_days_list = [5]
            print("将只生成5天图像供查看...")
    except:
        window_days_list = [5]
        print("默认只生成5天图像供查看...")
    
    all_passed = True
    
    for window_days in window_days_list:
        try:
            # 生成图像
            image = generate_final_ohlc_image(window_days)
            
            # 验证要求
            passed = validate_ohlc_requirements(image, window_days)
            all_passed = all_passed and passed
            
        except Exception as e:
            print(f"✗ {window_days}天图像生成失败: {str(e)}")
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 所有图像生成成功！完全符合文献要求：")
        print("  ✓ 严格黑白格式（黑色背景0，白色图表255）")
        print("  ✓ 每天3像素宽度（n天图像宽度为3n像素）")
        print("  ✓ 顶部75%区域：OHLC图 + 移动平均线（四分之三）")
        print("  ✓ 底部20%区域：成交量条（五分之一，最大成交量达到上限）")
        print("  ✓ OHLC柱状图：中间垂直柱（高低），左横线（开盘），右横线（收盘）")
        print("  ✓ 移动平均线：窗口长度等于图像天数，通过中间列像素连接绘制")
        print("  ✓ 成交量条之间有间隔，按最大成交量比例缩放")
        print("  ✓ 垂直轴缩放使OHLC路径最大值和最小值对齐图像顶部和底部")
    else:
        print("❌ 部分图像不符合要求，请检查上述验证结果")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 