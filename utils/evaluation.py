"""
模型评估模块
"""
import numpy as np
import pandas as pd
from utils.config import EVAL_CONFIG

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def calculate_portfolio_returns(self, predictions, returns, n_portfolios=10):
        """计算投资组合收益"""
        # 获取预测概率（涨的概率）
        pred_probs = predictions[:, 1]
        
        # 按预测概率排序，分为十分位组合
        sorted_indices = np.argsort(pred_probs)
        portfolio_size = len(predictions) // n_portfolios
        
        portfolio_returns = []
        
        for i in range(n_portfolios):
            start_idx = i * portfolio_size
            end_idx = (i + 1) * portfolio_size if i < n_portfolios - 1 else len(predictions)
            
            # 该组合的股票索引
            portfolio_indices = sorted_indices[start_idx:end_idx]
            
            # 计算等权重组合收益
            portfolio_return = np.mean([returns[idx] for idx in portfolio_indices])
            portfolio_returns.append(portfolio_return)
        
        return portfolio_returns
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        """计算夏普比率"""
        if len(returns) == 0:
            return 0.0
            
        excess_returns = np.array(returns) - risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
            
        # 年化夏普比率（假设252个交易日）
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        
        return sharpe
    
    def long_short_strategy(self, predictions, returns):
        """计算长短策略收益"""
        # 计算十分位组合收益
        portfolio_returns = self.calculate_portfolio_returns(predictions, returns)
        
        # H-L策略：做多最高十分位，做空最低十分位
        high_return = portfolio_returns[-1]  # 最高十分位
        low_return = portfolio_returns[0]    # 最低十分位
        
        long_short_return = high_return - low_return
        
        return long_short_return, portfolio_returns
    
    def evaluate_predictions(self, predictions, actual_returns, dates=None):
        """评估预测结果"""
        results = {}
        
        # 计算长短策略
        ls_return, portfolio_rets = self.long_short_strategy(predictions, actual_returns)
        
        # 计算各组合的夏普比率
        portfolio_sharpes = []
        for i, ret in enumerate(portfolio_rets):
            # 这里简化处理，实际应该用时间序列收益
            portfolio_sharpes.append(ret * np.sqrt(252))  # 简化年化
        
        # 长短组合夏普比率
        ls_sharpe = self.calculate_sharpe_ratio([ls_return])
        
        results = {
            'long_short_return': ls_return,
            'long_short_sharpe': ls_sharpe,
            'portfolio_returns': portfolio_rets,
            'portfolio_sharpes': portfolio_sharpes,
            'accuracy': self._calculate_accuracy(predictions, actual_returns)
        }
        
        return results
    
    def _calculate_accuracy(self, predictions, actual_returns):
        """计算预测准确率"""
        pred_labels = np.argmax(predictions, axis=1)
        actual_labels = [1 if ret > 0 else 0 for ret in actual_returns]
        
        accuracy = np.mean(pred_labels == actual_labels)
        return accuracy
    
    def print_evaluation_report(self, results, window_days):
        """打印评估报告"""
        print(f"\n=== {window_days}天模型评估结果 ===")
        print(f"预测准确率: {results['accuracy']:.4f}")
        print(f"长短策略收益: {results['long_short_return']:.4f}")
        print(f"长短策略夏普比率: {results['long_short_sharpe']:.4f}")
        
        print("\n十分位组合表现:")
        for i, (ret, sharpe) in enumerate(zip(results['portfolio_returns'], results['portfolio_sharpes'])):
            print(f"组合{i+1}: 收益={ret:.4f}, 夏普比率={sharpe:.4f}")
    
    def statistical_significance_test(self, sharpe_ratio, n_samples):
        """统计显著性检验（简化版）"""
        # 简化的t检验
        t_stat = sharpe_ratio * np.sqrt(n_samples)
        
        # 95%置信水平的临界值约为1.96
        is_significant = abs(t_stat) > 1.96
        
        return is_significant, t_stat 