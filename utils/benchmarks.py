"""
基准信号模块（论文要求：MOM、STR、WSTR、TREND）
"""
import numpy as np
import pandas as pd

class BenchmarkSignals:
    """基准信号计算器"""
    
    def __init__(self):
        self.signals = {}
    
    def calculate_momentum(self, returns, lookback_months=12, skip_months=1):
        """计算动量信号（MOM）
        
        Args:
            returns: 收益率序列
            lookback_months: 回望期（月）
            skip_months: 跳过的最近月数
        """
        # 2-12月动量（通常去掉最近1个月）
        if len(returns) < lookback_months:
            return np.nan
        
        # 跳过最近skip_months个月
        start_idx = skip_months
        end_idx = lookback_months
        
        momentum_return = np.prod(1 + returns[start_idx:end_idx]) - 1
        return momentum_return
    
    def calculate_short_term_reversal(self, returns, lookback_days=20):
        """计算短期反转信号（STR）
        
        Args:
            returns: 收益率序列
            lookback_days: 回望天数（1个月约20个交易日）
        """
        if len(returns) < lookback_days:
            return np.nan
        
        # 1个月短反转
        recent_returns = returns[-lookback_days:]
        str_signal = -np.mean(recent_returns)  # 负号表示反转
        return str_signal
    
    def calculate_weekly_short_term_reversal(self, returns, lookback_days=5):
        """计算周短期反转信号（WSTR）
        
        Args:
            returns: 收益率序列
            lookback_days: 回望天数（1周约5个交易日）
        """
        if len(returns) < lookback_days:
            return np.nan
        
        # 1周短反转
        recent_returns = returns[-lookback_days:]
        wstr_signal = -np.mean(recent_returns)  # 负号表示反转
        return wstr_signal
    
    def calculate_trend_signal(self, prices, short_window=20, medium_window=60, long_window=120):
        """计算趋势信号（TREND）
        
        基于Han, Zhou & Zhu (2016)的组合短/中/长期趋势信号
        
        Args:
            prices: 价格序列
            short_window: 短期窗口（天）
            medium_window: 中期窗口（天）
            long_window: 长期窗口（天）
        """
        if len(prices) < long_window:
            return np.nan
        
        # 计算不同周期的移动平均
        ma_short = np.mean(prices[-short_window:])
        ma_medium = np.mean(prices[-medium_window:])
        ma_long = np.mean(prices[-long_window:])
        
        # 当前价格
        current_price = prices[-1]
        
        # 趋势信号：组合短/中/长期趋势
        trend_signal = (
            0.5 * (current_price - ma_short) / ma_short +
            0.3 * (current_price - ma_medium) / ma_medium +
            0.2 * (current_price - ma_long) / ma_long
        )
        
        return trend_signal
    
    def calculate_all_benchmarks(self, prices, returns):
        """计算所有基准信号"""
        benchmarks = {}
        
        # 计算动量信号
        benchmarks['MOM'] = self.calculate_momentum(returns)
        
        # 计算短期反转信号
        benchmarks['STR'] = self.calculate_short_term_reversal(returns)
        
        # 计算周短期反转信号
        benchmarks['WSTR'] = self.calculate_weekly_short_term_reversal(returns)
        
        # 计算趋势信号
        benchmarks['TREND'] = self.calculate_trend_signal(prices)
        
        return benchmarks
    
    def create_benchmark_portfolios(self, stock_data, benchmark_type='MOM'):
        """创建基准投资组合
        
        Args:
            stock_data: 股票数据字典 {symbol: {'prices': [...], 'returns': [...]}}
            benchmark_type: 基准类型 ('MOM', 'STR', 'WSTR', 'TREND')
        """
        benchmark_signals = {}
        
        # 计算每只股票的基准信号
        for symbol, data in stock_data.items():
            if benchmark_type == 'MOM':
                signal = self.calculate_momentum(data['returns'])
            elif benchmark_type == 'STR':
                signal = self.calculate_short_term_reversal(data['returns'])
            elif benchmark_type == 'WSTR':
                signal = self.calculate_weekly_short_term_reversal(data['returns'])
            elif benchmark_type == 'TREND':
                signal = self.calculate_trend_signal(data['prices'])
            else:
                signal = np.nan
            
            benchmark_signals[symbol] = signal
        
        # 按信号排序，创建十分位组合
        valid_signals = {k: v for k, v in benchmark_signals.items() if not np.isnan(v)}
        
        if len(valid_signals) == 0:
            return None, None
        
        # 排序
        sorted_stocks = sorted(valid_signals.items(), key=lambda x: x[1])
        
        # 分为十分位
        n_stocks = len(sorted_stocks)
        decile_size = n_stocks // 10
        
        deciles = {}
        for i in range(10):
            start_idx = i * decile_size
            end_idx = (i + 1) * decile_size if i < 9 else n_stocks
            deciles[i] = [stock[0] for stock in sorted_stocks[start_idx:end_idx]]
        
        return deciles, benchmark_signals
    
    def evaluate_benchmark_performance(self, deciles, stock_data, holding_period_days=20):
        """评估基准策略表现"""
        if deciles is None:
            return None
        
        # 计算每个十分位的收益
        decile_returns = []
        
        for i in range(10):
            decile_stocks = deciles[i]
            if len(decile_stocks) == 0:
                decile_returns.append(0.0)
                continue
            
            # 计算等权重组合收益
            decile_return = 0.0
            valid_stocks = 0
            
            for stock in decile_stocks:
                if stock in stock_data and len(stock_data[stock]['returns']) > 0:
                    # 使用最近持有期的收益
                    recent_return = np.mean(stock_data[stock]['returns'][-holding_period_days:])
                    decile_return += recent_return
                    valid_stocks += 1
            
            if valid_stocks > 0:
                decile_return /= valid_stocks
            
            decile_returns.append(decile_return)
        
        # 计算H-L策略收益
        high_return = decile_returns[-1]  # 最高十分位
        low_return = decile_returns[0]    # 最低十分位
        long_short_return = high_return - low_return
        
        # 计算夏普比率
        if holding_period_days <= 7:
            frequency = 52
        elif holding_period_days <= 30:
            frequency = 12
        else:
            frequency = 4
        
        sharpe_ratio = long_short_return / np.std(decile_returns) * np.sqrt(frequency) if np.std(decile_returns) > 0 else 0
        
        return {
            'decile_returns': decile_returns,
            'long_short_return': long_short_return,
            'sharpe_ratio': sharpe_ratio,
            'frequency': frequency
        }
