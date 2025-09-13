"""
OHLC图像生成模块
"""
import numpy as np
from PIL import Image, ImageDraw
from utils.config import IMAGE_CONFIG
import pandas as pd

class OHLCImageGenerator:
    def __init__(self, window_days):
        self.config = IMAGE_CONFIG[window_days]
        self.width = self.config["width"]
        self.height = self.config["height"]
        self.days = self.config["days"]
        self.train_mean = None
        self.train_std = None
        
    def fit_normalizer(self, train_sequences):
        """计算训练集的均值和标准差
        
        Args:
            train_sequences: 训练集序列
        """
        # 生成所有训练图像
        train_images = []
        for seq in train_sequences:
            img = self.generate_image(seq)
            img = img.reshape(self.height, self.width, 1)
            train_images.append(img)
        
        train_images = np.array(train_images, dtype=np.float32)
        
        # 计算均值和标准差
        self.train_mean = np.mean(train_images)
        self.train_std = np.std(train_images)
        
        print(f"图像归一化参数：均值={self.train_mean:.3f}, 标准差={self.train_std:.3f}")
        
    def normalize_image(self, img):
        """使用训练集统计量归一化图像"""
        if self.train_mean is None or self.train_std is None:
            raise ValueError("请先调用fit_normalizer计算训练集统计量")
            
        # 标准化处理
        img = (img - self.train_mean) / (self.train_std + 1e-8)  # 添加小量防止除零
        return img

    def generate_image(self, df_window):
        """生成OHLC图像"""
        # 创建黑色背景图像
        img = Image.new('L', (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        
        # 简化区域划分（严格按照文献要求）
        volume_area_height = int(self.height * 0.2)   # 成交量区域占底部20%
        price_area_height = self.height - volume_area_height  # 价格区域占顶部80%
        
        # 获取移动平均线（窗口长度等于图像天数）
        ma_values = df_window['Adj_Close_calc'].rolling(window=self.days, min_periods=1).mean()
        
        # 统一的价格范围计算（包括OHLC和移动平均线）
        # 按文献要求：垂直轴以最高价/最低价为边界归一化
        all_prices = []
        
        # 收集所有非空的价格数据
        for col in ['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close_calc']:
            valid_prices = df_window[col].dropna()
            if len(valid_prices) > 0:
                all_prices.extend(valid_prices.values)
        
        # 添加移动平均线数据
        ma_valid = ma_values.dropna()
        if len(ma_valid) > 0:
            all_prices.extend(ma_valid.values)
        
        # 计算价格范围
        if len(all_prices) > 0:
            all_prices = np.array(all_prices)
            price_min, price_max = all_prices.min(), all_prices.max()
            price_range = price_max - price_min if price_max > price_min else 1
        else:
            # 如果没有有效价格数据，使用默认范围
            price_min, price_max = 0, 1
            price_range = 1
        
        # 获取成交量范围
        volume_max = df_window['Volume'].max()
        
        # 绘制每日OHLC
        for i, (_, row) in enumerate(df_window.iterrows()):
            x_base = i * 3  # 每天3像素宽
            
            # 按文献要求处理缺失数据
            if 'has_ohlc' in row and not row['has_ohlc']:
                # 完全缺失OHLC数据：对应像素列留空（全黑，像素值0）
                continue
            elif 'has_high_low' in row and not row['has_high_low']:
                # 缺失高低价数据：整列留空
                continue
            
            # 检查各个价格是否存在
            has_open = not pd.isna(row['Adj_Open'])
            has_high = not pd.isna(row['Adj_High'])
            has_low = not pd.isna(row['Adj_Low'])
            has_close = not pd.isna(row['Adj_Close_calc'])
            
            # 只有当高低价都存在时才能绘制
            if has_high and has_low:
                # 统一使用相同的价格范围缩放（最高价对应顶部，最低价对应底部）
                high_y = self._scale_price(row['Adj_High'], price_min, price_range, price_area_height)
                low_y = self._scale_price(row['Adj_Low'], price_min, price_range, price_area_height)
                
                # 绘制高低价垂直线（中间像素）
                draw.line([(x_base + 1, high_y), (x_base + 1, low_y)], fill=255, width=1)
                
                # 只有当开盘价存在时才绘制开盘价横线
                if has_open:
                    open_y = self._scale_price(row['Adj_Open'], price_min, price_range, price_area_height)
                    draw.line([(x_base, open_y), (x_base + 1, open_y)], fill=255, width=1)
                
                # 只有当收盘价存在时才绘制收盘价横线
                if has_close:
                    close_y = self._scale_price(row['Adj_Close_calc'], price_min, price_range, price_area_height)
                    draw.line([(x_base + 1, close_y), (x_base + 2, close_y)], fill=255, width=1)
            
            # 绘制成交量条（底部20%区域，最大成交量达到上限）
            if volume_max > 0 and not pd.isna(row['Volume']):
                volume_ratio = row['Volume'] / volume_max
                volume_height = int(volume_ratio * volume_area_height)
                volume_y = self.height - volume_height  # 从底部向上绘制
                
                # 中间1像素画成交量条，确保成交量条之间有间隔
                if volume_height > 0:
                    draw.rectangle([
                        (x_base + 1, volume_y), 
                        (x_base + 1, self.height - 1)
                    ], fill=255)
        
        # 绘制移动平均线（使用统一的价格范围）
        if len(ma_values) > 1:
            ma_points = []
            for i in range(len(ma_values)):
                x = i * 3 + 1  # 中间列位置
                ma_val = ma_values.iloc[i]
                
                # 处理NaN值（如果仍然存在）
                if np.isnan(ma_val):
                    ma_val = df_window['Adj_Close_calc'].iloc[i]
                
                # 使用统一的价格范围进行缩放
                y = self._scale_price(ma_val, price_min, price_range, price_area_height)
                ma_points.append((x, y))
            
            # 绘制连续的移动平均线
            for i in range(len(ma_points) - 1):
                draw.line([ma_points[i], ma_points[i + 1]], fill=255, width=1)
        
        return np.array(img)
    
    def _scale_price(self, price, price_min, price_range, area_height):
        """将价格缩放到图像坐标（统一的价格归一化）"""
        normalized = (price - price_min) / price_range
        # 翻转Y轴（图像坐标系）：最高价对应顶部(0)，最低价对应底部
        y = int((1 - normalized) * (area_height - 1))
        return max(0, min(area_height - 1, y))
    
    def _scale_ohlc_price(self, price, price_min, price_range, area_height, start_y):
        """将OHLC价格缩放到专用区域（保留用于兼容性）"""
        normalized = (price - price_min) / price_range
        # 翻转Y轴（图像坐标系）
        y = int((1 - normalized) * (area_height - 1)) + start_y
        return max(start_y, min(start_y + area_height - 1, y))
    
    def _scale_ma_price(self, price, ma_min, ma_range, area_height):
        """将移动平均线价格缩放到图像坐标（保留用于兼容性）"""
        normalized = (price - ma_min) / ma_range
        # 翻转Y轴（图像坐标系）
        y = int((1 - normalized) * (area_height - 1))
        return max(0, min(area_height - 1, y))
    
    def generate_batch(self, sequences):
        """批量生成图像"""
        images = []
        for seq in sequences:
            img = self.generate_image(seq)
            # 添加通道维度
            img = img.reshape(self.height, self.width, 1)
            
            # 如果已经计算了训练集统计量，进行标准化
            if self.train_mean is not None and self.train_std is not None:
                img = self.normalize_image(img)
            else:
                # 临时保持原有的简单归一化，但打印警告
                print("警告：未计算训练集统计量，使用简单归一化")
                img = img / 255.0
                
            images.append(img)
        
        return np.array(images, dtype=np.float32) 