"""
CNN股票预测系统演示程序（无需TensorFlow）
展示数据处理和OHLC图像生成功能
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# 简化的配置
IMAGE_CONFIG = {
    5: {"width": 15, "height": 64, "days": 5},
    20: {"width": 60, "height": 64, "days": 20},
    60: {"width": 180, "height": 64, "days": 60}
}

# 图像保存路径
IMAGE_SAVE_PATH = r"C:\Users\Anny\PycharmProjects\CNN\test_images"

class SimpleStockProcessor:
    """简化的股票数据处理器"""
    
    def __init__(self):
        self.data = {}
    
    def load_data(self):
        """加载股票数据"""
        data_dir = Path("data")
        for file in data_dir.glob("*.csv"):
            symbol = self._extract_symbol(file.name)
            try:
                df = pd.read_csv(file, parse_dates=['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                self.data[symbol] = self._clean_data(df)
                print(f"✓ 加载 {symbol}: {len(df)} 条记录")
            except Exception as e:
                print(f"✗ 加载 {file.name} 失败: {e}")
        return self.data
    
    def _extract_symbol(self, filename):
        """从文件名提取股票代码"""
        if "GSPC" in filename:
            return "SPX"
        elif "IXIC" in filename:
            return "NASDAQ" 
        elif "DJI" in filename:
            return "DOW"
        return filename.split()[0]
    
    def _clean_data(self, df):
        """数据清洗"""
        df = df.dropna().copy()
        df['Return'] = df['Close'].pct_change()
        
        # CRSP调整价格
        adj_returns = df['Adj Close'].pct_change().fillna(0)
        adj_prices = np.zeros(len(df))
        adj_prices[0] = 1.0
        
        for i in range(1, len(df)):
            adj_prices[i] = adj_prices[i-1] * (1 + adj_returns.iloc[i])
        
        price_ratio = adj_prices / df['Close'].values
        df['Adj_Open'] = df['Open'] * price_ratio
        df['Adj_High'] = df['High'] * price_ratio
        df['Adj_Low'] = df['Low'] * price_ratio
        df['Adj_Close_calc'] = adj_prices
        
        return df

class SimpleImageGenerator:
    """简化的OHLC图像生成器"""
    
    def __init__(self, window_days):
        self.config = IMAGE_CONFIG[window_days]
        self.width = self.config["width"]
        self.height = self.config["height"]
        self.days = self.config["days"]
    
    def generate_image(self, df_window):
        """生成OHLC图像"""
        img = Image.new('L', (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        
        # 价格区域80%，成交量区域20%
        price_area_height = int(self.height * 0.8)
        
        # 价格范围计算
        all_prices = np.concatenate([
            df_window['Adj_Open'].values,
            df_window['Adj_High'].values,
            df_window['Adj_Low'].values,
            df_window['Adj_Close_calc'].values
        ])
        price_min, price_max = all_prices.min(), all_prices.max()
        price_range = price_max - price_min if price_max > price_min else 1
        
        volume_max = df_window['Volume'].max()
        
        # 移动平均线
        ma_values = df_window['Adj_Close_calc'].rolling(window=len(df_window)).mean()
        
        # 绘制OHLC
        for i, (_, row) in enumerate(df_window.iterrows()):
            x_base = i * 3
            
            # 价格缩放
            open_y = self._scale_price(row['Adj_Open'], price_min, price_range, price_area_height)
            high_y = self._scale_price(row['Adj_High'], price_min, price_range, price_area_height)
            low_y = self._scale_price(row['Adj_Low'], price_min, price_range, price_area_height)
            close_y = self._scale_price(row['Adj_Close_calc'], price_min, price_range, price_area_height)
            
            # 绘制高低价垂直线
            draw.line([(x_base + 1, high_y), (x_base + 1, low_y)], fill=255, width=1)
            # 开盘价横线
            draw.line([(x_base, open_y), (x_base + 1, open_y)], fill=255, width=1)
            # 收盘价横线
            draw.line([(x_base + 1, close_y), (x_base + 2, close_y)], fill=255, width=1)
            
            # 成交量条
            if volume_max > 0:
                volume_height = int((row['Volume'] / volume_max) * (self.height - price_area_height))
                volume_y = self.height - volume_height
                draw.rectangle([(x_base, volume_y), (x_base + 2, self.height - 1)], fill=255)
        
        # 绘制移动平均线
        ma_points = []
        for i, ma_val in enumerate(ma_values.dropna()):
            x = i * 3 + 1
            y = self._scale_price(ma_val, price_min, price_range, price_area_height)
            ma_points.append((x, y))
        
        for i in range(len(ma_points) - 1):
            draw.line([ma_points[i], ma_points[i + 1]], fill=255, width=1)
        
        return np.array(img)
    
    def _scale_price(self, price, price_min, price_range, area_height):
        """价格缩放到图像坐标"""
        normalized = (price - price_min) / price_range
        y = int((1 - normalized) * (area_height - 1))
        return max(0, min(area_height - 1, y))

def create_sample_sequences(df, window_days):
    """创建样本序列"""
    sequences = []
    for i in range(len(df) - window_days + 1):
        seq = df.iloc[i:i+window_days].copy()
        sequences.append(seq)
    return sequences

def save_demo_images():
    """保存演示图像"""
    print("\n=== CNN股票预测系统演示 ===")
    
    # 加载数据
    processor = SimpleStockProcessor()
    data = processor.load_data()
    
    if not data:
        print("❌ 未找到数据文件，请确保data目录下有CSV文件")
        return
    
    # 选择一个数据集进行演示
    symbol = list(data.keys())[0]
    df = data[symbol]
    print(f"\n📊 使用 {symbol} 数据进行演示，共 {len(df)} 条记录")
    print(f"📅 时间范围: {df['Date'].min()} 至 {df['Date'].max()}")
    
    # 创建输出目录
    os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
    
    # 为不同时间窗口生成图像
    for window_days in [5, 20, 60]:
        print(f"\n🖼️  生成 {window_days} 天OHLC图像...")
        
        # 创建序列
        sequences = create_sample_sequences(df, window_days)
        if len(sequences) < 100:
            print(f"   ⚠️  数据不足，仅有 {len(sequences)} 个序列")
            continue
        
        # 生成图像
        generator = SimpleImageGenerator(window_days)
        
        # 选择几个有代表性的序列
        sample_indices = [0, len(sequences)//4, len(sequences)//2, 3*len(sequences)//4, -1]
        
        fig, axes = plt.subplots(1, len(sample_indices), figsize=(20, 4))
        if len(sample_indices) == 1:
            axes = [axes]
        
        for i, seq_idx in enumerate(sample_indices):
            seq = sequences[seq_idx]
            img = generator.generate_image(seq)
            
            axes[i].imshow(img, cmap='gray', aspect='auto')
            axes[i].set_title(f'序列 {seq_idx+1}\n{seq.iloc[0]["Date"].strftime("%Y-%m-%d")}')
            axes[i].set_xlabel('时间（像素）')
            if i == 0:
                axes[i].set_ylabel('价格+成交量')
        
        plt.suptitle(f'{symbol} - {window_days}天OHLC图像示例', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{IMAGE_SAVE_PATH}/{symbol}_{window_days}days_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ 保存图像: {IMAGE_SAVE_PATH}/{symbol}_{window_days}days_samples.png")
        print(f"   📏 图像尺寸: {generator.width}×{generator.height} 像素")
    
    print(f"\n🎉 演示完成！请查看 {IMAGE_SAVE_PATH} 目录下的图像文件")
    print("   图像特征：")
    print("   • 白色线条，黑色背景")
    print("   • 每天3像素宽：开盘+高低+收盘")
    print("   • 底部20%为成交量条")
    print("   • 白色移动平均线叠加")

if __name__ == "__main__":
    save_demo_images() 