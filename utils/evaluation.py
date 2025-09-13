"""
模型评估模块
"""
import numpy as np
import pandas as pd
from utils.config import EVAL_CONFIG

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def calculate_portfolio_returns(self, predictions, returns, market_caps=None, n_portfolios=10, weighting='equal'):
        """计算投资组合收益（论文要求：等权/价值加权）
        
        Args:
            predictions: 预测概率
            returns: 实际收益率
            market_caps: 市值数据（用于价值加权）
            n_portfolios: 组合数量（论文要求10个十分位）
            weighting: 加权方式 ('equal' 或 'value')
        """
        # 获取预测概率（涨的概率）
        if predictions.ndim == 2:
            pred_probs = predictions[:, 1]  # 二分类概率
        else:
            pred_probs = predictions  # 一维概率
        
        # 按预测概率排序，分为十分位组合
        sorted_indices = np.argsort(pred_probs)
        portfolio_size = len(predictions) // n_portfolios
        
        portfolio_returns = []
        portfolio_weights = []
        
        for i in range(n_portfolios):
            start_idx = i * portfolio_size
            end_idx = (i + 1) * portfolio_size if i < n_portfolios - 1 else len(predictions)
            
            # 该组合的股票索引
            portfolio_indices = sorted_indices[start_idx:end_idx]
            
            if weighting == 'equal':
                # 等权重组合
                weights = np.ones(len(portfolio_indices)) / len(portfolio_indices)
                portfolio_return = np.mean([returns[idx] for idx in portfolio_indices])
            elif weighting == 'value' and market_caps is not None:
                # 价值加权组合（按市值加权）
                portfolio_market_caps = [market_caps[idx] for idx in portfolio_indices]
                weights = np.array(portfolio_market_caps) / np.sum(portfolio_market_caps)
                portfolio_return = np.sum([returns[idx] * weights[j] for j, idx in enumerate(portfolio_indices)])
            else:
                # 默认等权重
                weights = np.ones(len(portfolio_indices)) / len(portfolio_indices)
                portfolio_return = np.mean([returns[idx] for idx in portfolio_indices])
            
            portfolio_returns.append(portfolio_return)
            portfolio_weights.append(weights)
        
        return portfolio_returns, portfolio_weights
    
    def calculate_sharpe_ratio(self, returns, frequency=252, risk_free_rate=0.0):
        """计算年化夏普比率（论文要求）
        
        Args:
            returns: 收益率序列
            frequency: 年化频率（weekly=52, monthly=12, quarterly=4）
            risk_free_rate: 无风险利率（论文中未减去，设为0）
        """
        if len(returns) == 0:
            return 0.0
            
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
            
        # 论文公式：Sharpe = mean(period_returns) / std(period_returns) * sqrt(f)
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(frequency)
        
        return sharpe
    
    def long_short_strategy(self, predictions, returns, market_caps=None, weighting='equal'):
        """计算H-L长短策略收益（论文要求）
        
        Args:
            predictions: 预测概率
            returns: 实际收益率
            market_caps: 市值数据
            weighting: 加权方式
        """
        # 计算十分位组合收益
        portfolio_returns, portfolio_weights = self.calculate_portfolio_returns(
            predictions, returns, market_caps, weighting=weighting
        )
        
        # H-L策略：做多最高十分位(decile10)，做空最低十分位(decile1)
        high_return = portfolio_returns[-1]  # 最高十分位
        low_return = portfolio_returns[0]    # 最低十分位
        
        long_short_return = high_return - low_return
        
        return long_short_return, portfolio_returns, portfolio_weights
    
    def calculate_turnover(self, weights_t, weights_t1, returns_t1, holding_period_months=1):
        """计算换手率（论文公式）
        
        Args:
            weights_t: t时刻的权重
            weights_t1: t+1时刻的权重  
            returns_t1: t+1时刻的收益率
            holding_period_months: 持有期月数
        """
        # 论文公式：Turnover = (1/M) * (1/T) * sum_{t=1..T} sum_i | w_{i,t+1} - w_{i,t} * (1 + r_{i,t+1}) / (1 + sum_j w_{j,t} r_{j,t+1}) |
        # 其中 M = 持有期月数，T = 期数
        
        if len(weights_t) != len(weights_t1) or len(weights_t) != len(returns_t1):
            return 0.0
        
        # 计算组合收益率
        portfolio_return = np.sum([w * r for w, r in zip(weights_t, returns_t1)])
        
        # 计算调整后的权重
        adjusted_weights = []
        for w, r in zip(weights_t, returns_t1):
            adjusted_w = w * (1 + r) / (1 + portfolio_return)
            adjusted_weights.append(adjusted_w)
        
        # 计算换手率
        turnover = np.sum([abs(w1 - w_adj) for w1, w_adj in zip(weights_t1, adjusted_weights)])
        
        # 按持有期调整
        monthly_turnover = turnover / holding_period_months
        
        return monthly_turnover
    
    def calculate_cumulative_returns(self, returns):
        """计算累计净值曲线"""
        cumulative_returns = np.cumprod(1 + np.array(returns))
        return cumulative_returns
    
    def evaluate_predictions(self, predictions, actual_returns, dates=None, market_caps=None, holding_period_days=20):
        """评估预测结果（论文要求：等权/价值加权、换手率、净值曲线）"""
        results = {}
        
        # 计算年化频率
        if holding_period_days <= 7:
            frequency = 52  # 周策略
        elif holding_period_days <= 30:
            frequency = 12  # 月策略
        else:
            frequency = 4   # 季策略
        
        # 计算等权重策略
        ls_return_equal, portfolio_rets_equal, portfolio_weights_equal = self.long_short_strategy(
            predictions, actual_returns, weighting='equal'
        )
        
        # 计算价值加权策略（如果有市值数据）
        if market_caps is not None:
            ls_return_value, portfolio_rets_value, portfolio_weights_value = self.long_short_strategy(
                predictions, actual_returns, market_caps, weighting='value'
            )
        else:
            ls_return_value, portfolio_rets_value, portfolio_weights_value = None, None, None
        
        # 计算夏普比率
        ls_sharpe_equal = self.calculate_sharpe_ratio([ls_return_equal], frequency)
        if ls_return_value is not None:
            ls_sharpe_value = self.calculate_sharpe_ratio([ls_return_value], frequency)
        else:
            ls_sharpe_value = None
        
        # 计算换手率（简化版，需要时间序列数据）
        turnover_equal = 0.0  # 需要实际的时间序列权重数据
        turnover_value = 0.0
        
        # 计算累计净值曲线
        cumulative_equal = self.calculate_cumulative_returns([ls_return_equal])
        if ls_return_value is not None:
            cumulative_value = self.calculate_cumulative_returns([ls_return_value])
        else:
            cumulative_value = None
        
        results = {
            # 等权重结果
            'equal_weight': {
                'long_short_return': ls_return_equal,
                'long_short_sharpe': ls_sharpe_equal,
                'portfolio_returns': portfolio_rets_equal,
                'portfolio_weights': portfolio_weights_equal,
                'turnover': turnover_equal,
                'cumulative_returns': cumulative_equal
            },
            # 价值加权结果
            'value_weight': {
                'long_short_return': ls_return_value,
                'long_short_sharpe': ls_sharpe_value,
                'portfolio_returns': portfolio_rets_value,
                'portfolio_weights': portfolio_weights_value,
                'turnover': turnover_value,
                'cumulative_returns': cumulative_value
            },
            # 其他指标
            'accuracy': self._calculate_accuracy(predictions, actual_returns),
            'frequency': frequency,
            'holding_period_days': holding_period_days
        }
        
        return results
    
    def _calculate_accuracy(self, predictions, actual_returns):
        """计算预测准确率"""
        pred_labels = np.argmax(predictions, axis=1)
        actual_labels = [1 if ret > 0 else 0 for ret in actual_returns]
        
        accuracy = np.mean(pred_labels == actual_labels)
        return accuracy
    
    def print_evaluation_report(self, results, window_days):
        """打印评估报告（论文要求格式）"""
        print(f"\n=== {window_days}天模型评估结果 ===")
        print(f"预测准确率: {results['accuracy']:.4f}")
        print(f"持有期: {results['holding_period_days']}天")
        print(f"年化频率: {results['frequency']}")
        
        # 等权重结果
        print(f"\n--- 等权重组合 ---")
        eq_results = results['equal_weight']
        print(f"H-L策略收益: {eq_results['long_short_return']:.4f}")
        print(f"H-L策略夏普比率: {eq_results['long_short_sharpe']:.4f}")
        print(f"月度换手率: {eq_results['turnover']:.4f}")
        
        print("\n十分位组合表现（等权重）:")
        for i, ret in enumerate(eq_results['portfolio_returns']):
            print(f"组合{i+1}: 收益={ret:.4f}")
        
        # 价值加权结果（如果有）
        if results['value_weight']['long_short_return'] is not None:
            print(f"\n--- 价值加权组合 ---")
            val_results = results['value_weight']
            print(f"H-L策略收益: {val_results['long_short_return']:.4f}")
            print(f"H-L策略夏普比率: {val_results['long_short_sharpe']:.4f}")
            print(f"月度换手率: {val_results['turnover']:.4f}")
            
            print("\n十分位组合表现（价值加权）:")
            for i, ret in enumerate(val_results['portfolio_returns']):
                print(f"组合{i+1}: 收益={ret:.4f}")
        else:
            print(f"\n--- 价值加权组合 ---")
            print("无市值数据，跳过价值加权分析")
    
    def statistical_significance_test(self, sharpe_ratio, n_samples):
        """统计显著性检验（简化版）"""
        # 简化的t检验
        t_stat = sharpe_ratio * np.sqrt(n_samples)
        
        # 95%置信水平的临界值约为1.96
        is_significant = abs(t_stat) > 1.96
        
        return is_significant, t_stat 