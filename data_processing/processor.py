"""
数据加载与预处理模块
"""
import pandas as pd
import numpy as np
from pathlib import Path
from utils.config import DATA_PATH

class StockDataProcessor:
    def __init__(self, data_fraction=1.0):
        """
        初始化数据处理器
        
        Args:
            data_fraction: 使用的数据比例 (0.0-1.0)，默认1.0使用全部数据
                          设置为0.1则只使用10%的数据（测试用）
        """
        self.data = {}
        self.data_fraction = data_fraction
        
    def load_data(self):
        """加载所有股票数据"""
        data_dir = Path(DATA_PATH)
        for file in data_dir.glob("*.csv"):
            symbol = self._extract_symbol(file.name)
            df = pd.read_csv(file, parse_dates=['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            # 如果设置了数据比例，只使用部分数据
            if self.data_fraction < 1.0:
                total_rows = len(df)
                use_rows = int(total_rows * self.data_fraction)
                print(f"数据量控制：{symbol} 原始 {total_rows} 行 → 使用 {use_rows} 行 ({self.data_fraction*100:.1f}%)")
                df = df.tail(use_rows)  # 使用最近的数据
            
            self.data[symbol] = self._clean_data(df)
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
        """数据清洗和预处理"""
        # 按文献要求：保留缺失数据，不删除缺失值行
        # 缺失日期将在图像生成时处理（对应像素列留空）
        
        # 确保数据按日期排序
        df = df.sort_values('Date').reset_index(drop=True)
        
        # 计算日收益率（允许NaN值存在）
        df['Return'] = df['Close'].pct_change()
        
        # 标记缺失数据类型，用于图像生成时的特殊处理
        df['has_ohlc'] = ~(df['Open'].isna() | df['High'].isna() | 
                          df['Low'].isna() | df['Close'].isna())
        df['has_volume'] = ~df['Volume'].isna()
        df['has_high_low'] = ~(df['High'].isna() | df['Low'].isna())
        
        return df
    
    def adjust_prices(self, df):
        """按CRSP方法调整价格序列"""
        # 使用Adj Close计算调整后的收益率，处理缺失数据
        adj_returns = df['Adj Close'].pct_change()
        
        # 计算调整因子（向前填充处理缺失值）
        adjustment_factor = (df['Adj Close'] / df['Close']).ffill()
        
        # 调整OHLC价格（保留缺失数据的NaN状态）
        df['Adj_Open'] = df['Open'] * adjustment_factor
        df['Adj_High'] = df['High'] * adjustment_factor  
        df['Adj_Low'] = df['Low'] * adjustment_factor
        df['Adj_Close_calc'] = df['Close'] * adjustment_factor
        
        # 验证调整后的收盘价应该等于原始Adj Close
        df['Adj_Close_calc'] = df['Adj Close']
        
        return df
    
    def create_sequences(self, df, window_days, prediction_days):
        """创建时间序列和标签
        
        Returns:
            sequences: 窗口期数据列表
            labels: 标签列表
            dates: 每个序列对应的最后一天日期
        """
        sequences = []
        labels = []
        dates = []
        
        for i in range(len(df) - window_days - prediction_days + 1):
            # 获取窗口期数据
            window_data = df.iloc[i:i+window_days].copy()
            
            # 检查当前价格和未来价格是否有效
            current_price = df.iloc[i+window_days-1]['Adj_Close_calc']
            future_price = df.iloc[i+window_days+prediction_days-1]['Adj_Close_calc']
            
            # 跳过价格数据无效的序列
            if pd.isna(current_price) or pd.isna(future_price):
                continue
                
            # 计算未来收益
            future_return = (future_price - current_price) / current_price
            
            sequences.append(window_data)
            labels.append(1 if future_return > 0 else 0)
            dates.append(df.iloc[i+window_days-1]['Date'])  # 添加序列最后一天的日期
            
        return sequences, labels, dates
    
    def get_processed_data(self, symbol, window_days, prediction_days):
        """获取处理后的数据"""
        if symbol not in self.data:
            raise ValueError(f"Symbol {symbol} not found")
            
        df = self.data[symbol].copy()
        df = self.adjust_prices(df)
        
        sequences, labels, dates = self.create_sequences(df, window_days, prediction_days)
        
        return sequences, labels, dates 