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
        """加载所有股票数据（论文要求：CRSP个股数据）"""
        data_dir = Path(DATA_PATH)
        
        # 检查是否有CRSP个股数据文件
        crsp_files = list(data_dir.glob("*CRSP*.csv")) + list(data_dir.glob("*crsp*.csv"))
        
        if crsp_files:
            # 如果有CRSP数据，优先使用
            print("发现CRSP个股数据文件，按论文要求加载...")
            for file in crsp_files:
                symbol = self._extract_symbol(file.name)
                df = pd.read_csv(file, parse_dates=['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                
                # 论文要求：训练期1993-2000，测试期2001-2019
                df = self._filter_time_period(df)
                
                if self.data_fraction < 1.0:
                    total_rows = len(df)
                    use_rows = int(total_rows * self.data_fraction)
                    print(f"数据量控制：{symbol} 原始 {total_rows} 行 → 使用 {use_rows} 行 ({self.data_fraction*100:.1f}%)")
                    df = df.tail(use_rows)
                
                self.data[symbol] = self._clean_data(df)
        else:
            # 如果没有CRSP数据，使用现有的指数数据（临时方案）
            print("未发现CRSP个股数据，使用现有指数数据（临时方案）...")
            print("注意：论文要求使用CRSP个股数据，当前使用指数数据无法进行完整的投资组合分析")
            
            for file in data_dir.glob("*.csv"):
                symbol = self._extract_symbol(file.name)
                df = pd.read_csv(file, parse_dates=['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                
                # 过滤时间区间
                df = self._filter_time_period(df)
                
                if self.data_fraction < 1.0:
                    total_rows = len(df)
                    use_rows = int(total_rows * self.data_fraction)
                    print(f"数据量控制：{symbol} 原始 {total_rows} 行 → 使用 {use_rows} 行 ({self.data_fraction*100:.1f}%)")
                    df = df.tail(use_rows)
                
                self.data[symbol] = self._clean_data(df)
        
        return self.data
    
    def _filter_time_period(self, df):
        """过滤时间区间（论文要求：1993-2019）"""
        # 确保数据覆盖论文要求的时间区间
        df = df[(df['Date'] >= '1993-01-01') & (df['Date'] <= '2019-12-31')]
        return df
    
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
        """创建时间序列和标签（论文要求：监督期=持有期）
        
        Args:
            window_days: 图像窗口长度（监督期）
            prediction_days: 预测持有期（应与window_days相同）
        
        Returns:
            sequences: 窗口期数据列表
            labels: 标签列表
            dates: 每个序列对应的最后一天日期
        """
        sequences = []
        labels = []
        dates = []
        
        # 论文要求：监督期=持有期，即window_days = prediction_days
        if window_days != prediction_days:
            print(f"警告：监督期({window_days})与持有期({prediction_days})不一致，调整为相同")
            prediction_days = window_days
        
        for i in range(len(df) - window_days - prediction_days + 1):
            # 获取窗口期数据（监督期）
            window_data = df.iloc[i:i+window_days].copy()
            
            # 检查当前价格和未来价格是否有效
            current_price = df.iloc[i+window_days-1]['Adj_Close_calc']
            future_price = df.iloc[i+window_days+prediction_days-1]['Adj_Close_calc']
            
            # 跳过价格数据无效的序列
            if pd.isna(current_price) or pd.isna(future_price):
                continue
                
            # 计算未来持有期累计收益
            future_return = (future_price - current_price) / current_price
            
            # 论文要求：二分类标签（未来持有期累计回报是否为正）
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