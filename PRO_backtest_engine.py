"""
PROFESSIONAL BACKTEST ENGINE
Institutional-grade backtesting with:
- Transaction costs
- Slippage modeling
- Position limits and risk management
- Multiple performance metrics
- Statistical significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for backtest"""
    initial_capital: float = 100000
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage_bps: float = 2.0  # 2 basis points slippage
    max_leverage: float = 1.5
    max_position_size: float = 0.40
    min_position_size: float = -0.20
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    risk_free_rate: float = 0.02


class ProfessionalBacktestEngine:
    """
    Institutional-grade backtest engine
    Handles realistic trading costs, position limits, and comprehensive metrics
    """
    
    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtest engine
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        self.trades = []
        self.portfolio_value = []
        self.positions_history = []
        
        print("Professional Backtest Engine initialized")
        print(f"  Initial capital: ${self.config.initial_capital:,.0f}")
        print(f"  Transaction cost: {self.config.transaction_cost:.2%}")
        print(f"  Slippage: {self.config.slippage_bps} bps")
    
    def run_backtest(self,
                     positions: pd.DataFrame,
                     prices: pd.DataFrame,
                     regime: pd.Series = None) -> Dict:
        """
        Run comprehensive backtest
        
        Args:
            positions: DataFrame of position sizes (target weights)
            prices: DataFrame of asset prices
            regime: Optional regime series for regime-based analysis
            
        Returns:
            Dictionary with backtest results
        """
        print("\n" + "="*80)
        print("RUNNING PROFESSIONAL BACKTEST")
        print("="*80)
        
        # Align data
        common_dates = positions.index.intersection(prices.index)
        positions = positions.loc[common_dates]
        prices = prices.loc[common_dates]
        
        # Initialize tracking
        n_days = len(common_dates)
        n_assets = len(prices.columns)
        
        portfolio_values = np.zeros(n_days)
        cash = np.zeros(n_days)
        holdings = np.zeros((n_days, n_assets))
        
        # Starting values
        portfolio_values[0] = self.config.initial_capital
        cash[0] = self.config.initial_capital
        
        print(f"Backtesting {n_days} days, {n_assets} assets...")
        
        # Simulate trading
        for i in range(1, n_days):
            date = common_dates[i]
            prev_date = common_dates[i-1]
            
            # Current prices
            current_prices = prices.loc[date].values
            prev_prices = prices.loc[prev_date].values
            
            # Update holdings value from price changes
            holdings_value = holdings[i-1] * current_prices
            current_portfolio_value = cash[i-1] + holdings_value.sum()
            
            # Target positions
            target_positions = positions.loc[date].values
            
            # Apply position limits
            target_positions = np.clip(
                target_positions,
                self.config.min_position_size,
                self.config.max_position_size
            )
            
            # Apply leverage limit
            gross_exposure = np.abs(target_positions).sum()
            if gross_exposure > self.config.max_leverage:
                target_positions = target_positions * (self.config.max_leverage / gross_exposure)
            
            # Target holdings in shares
            target_dollar_positions = target_positions * current_portfolio_value
            target_holdings = target_dollar_positions / current_prices
            
            # Calculate trades needed
            trades = target_holdings - holdings[i-1]
            
            # Apply transaction costs and slippage
            total_trade_value = np.abs(trades * current_prices).sum()
            transaction_costs = total_trade_value * self.config.transaction_cost
            
            # Slippage (proportional to trade size)
            slippage_cost = total_trade_value * (self.config.slippage_bps / 10000)
            
            # Execute trades
            holdings[i] = holdings[i-1] + trades
            cash[i] = cash[i-1] - (trades * current_prices).sum() - transaction_costs - slippage_cost
            
            # Update portfolio value
            portfolio_values[i] = cash[i] + (holdings[i] * current_prices).sum()
            
            # Record significant trades
            if total_trade_value > current_portfolio_value * 0.05:  # >5% turnover
                self.trades.append({
                    'date': date,
                    'trade_value': total_trade_value,
                    'transaction_costs': transaction_costs,
                    'slippage': slippage_cost,
                    'portfolio_value': portfolio_values[i]
                })
        
        # Calculate returns
        portfolio_returns = pd.Series(
            np.diff(portfolio_values) / portfolio_values[:-1],
            index=common_dates[1:]
        )
        
        # Store results
        results = {
            'portfolio_values': pd.Series(portfolio_values, index=common_dates),
            'returns': portfolio_returns,
            'positions': positions,
            'trades': self.trades,
            'holdings': pd.DataFrame(holdings, index=common_dates, columns=prices.columns),
            'cash': pd.Series(cash, index=common_dates)
        }
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(results, regime)
        results['metrics'] = metrics
        
        # Statistical tests
        results['statistical_tests'] = self._run_statistical_tests(portfolio_returns)
        
        print("\n✓ Backtest complete!")
        self._print_summary(metrics)
        
        return results
    
    def _calculate_comprehensive_metrics(self, results: Dict, regime: pd.Series = None) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        returns = results['returns']
        portfolio_values = results['portfolio_values']
        
        metrics = {}
        
        # Basic returns
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        n_years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        
        metrics['total_return'] = total_return
        metrics['annualized_return'] = annualized_return
        metrics['n_years'] = n_years
        
        # Risk metrics
        metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        metrics['downside_deviation'] = returns[returns < 0].std() * np.sqrt(252)
        
        # Risk-adjusted returns
        metrics['sharpe_ratio'] = (annualized_return - self.config.risk_free_rate) / metrics['annualized_volatility']
        metrics['sortino_ratio'] = (annualized_return - self.config.risk_free_rate) / metrics['downside_deviation'] if metrics['downside_deviation'] > 0 else 0
        
        # Drawdown analysis
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        metrics['avg_drawdown'] = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
        
        # Calmar ratio
        metrics['calmar_ratio'] = annualized_return / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Win rate and profit factor
        winning_days = (returns > 0).sum()
        total_days = len(returns)
        metrics['win_rate'] = winning_days / total_days
        
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        metrics['profit_factor'] = gross_profits / gross_losses if gross_losses > 0 else np.inf
        
        # Trading activity
        metrics['n_trades'] = len(self.trades)
        total_trade_value = sum(t['trade_value'] for t in self.trades)
        metrics['avg_trade_size'] = total_trade_value / len(self.trades) if self.trades else 0
        metrics['total_transaction_costs'] = sum(t['transaction_costs'] for t in self.trades)
        metrics['total_slippage'] = sum(t['slippage'] for t in self.trades)
        metrics['turnover'] = total_trade_value / self.config.initial_capital
        
        # Statistical
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurt()
        
        # VaR and CVaR
        metrics['var_95'] = returns.quantile(0.05)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        metrics['var_99'] = returns.quantile(0.01)
        metrics['cvar_99'] = returns[returns <= metrics['var_99']].mean()
        
        # Regime-based metrics
        if regime is not None:
            regime_aligned = regime.reindex(returns.index)
            for r in regime_aligned.unique():
                if r >= 0:
                    regime_returns = returns[regime_aligned == r]
                    if len(regime_returns) > 0:
                        metrics[f'return_regime_{int(r)}'] = regime_returns.mean() * 252
                        metrics[f'sharpe_regime_{int(r)}'] = (regime_returns.mean() / regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0
        
        return metrics
    
    def _run_statistical_tests(self, returns: pd.Series) -> Dict:
        """Run statistical significance tests"""
        
        from scipy import stats
        
        tests = {}
        
        # T-test: Is mean return significantly different from zero?
        t_stat, p_value = stats.ttest_1samp(returns.dropna(), 0)
        tests['mean_return_ttest'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Normality test
        _, p_normal = stats.normaltest(returns.dropna())
        tests['normality_test'] = {
            'p_value': p_normal,
            'is_normal': p_normal > 0.05
        }
        
        # Autocorrelation (momentum/mean reversion)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(returns.dropna(), lags=[1, 5, 10], return_df=True)
        tests['autocorrelation'] = {
            'lag_1_pvalue': lb_test.loc[1, 'lb_pvalue'],
            'significant_autocorr': lb_test['lb_pvalue'].min() < 0.05
        }
        
        return tests
    
    def _print_summary(self, metrics: Dict):
        """Print backtest summary"""
        
        print("\n" + "="*80)
        print("BACKTEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"\nReturns:")
        print(f"  Total Return:        {metrics['total_return']:>10.2%}")
        print(f"  Annualized Return:   {metrics['annualized_return']:>10.2%}")
        print(f"  Annualized Vol:      {metrics['annualized_volatility']:>10.2%}")
        
        print(f"\nRisk-Adjusted:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")
        
        print(f"\nDrawdown:")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        print(f"  Avg Drawdown:        {metrics['avg_drawdown']:>10.2%}")
        
        print(f"\nTrading:")
        print(f"  Win Rate:            {metrics['win_rate']:>10.2%}")
        print(f"  Profit Factor:       {metrics['profit_factor']:>10.2f}")
        print(f"  Number of Trades:    {metrics['n_trades']:>10.0f}")
        print(f"  Total Costs:         ${metrics['total_transaction_costs'] + metrics['total_slippage']:>10,.2f}")
    
    def compare_strategies(self, results_list: List[Dict], names: List[str]) -> pd.DataFrame:
        """
        Compare multiple strategy results
        
        Args:
            results_list: List of backtest results
            names: List of strategy names
            
        Returns:
            Comparison DataFrame
        """
        comparison = []
        
        for result, name in zip(results_list, names):
            metrics = result['metrics']
            comparison.append({
                'Strategy': name,
                'Total Return': metrics['total_return'],
                'Annual Return': metrics['annualized_return'],
                'Volatility': metrics['annualized_volatility'],
                'Sharpe': metrics['sharpe_ratio'],
                'Sortino': metrics['sortino_ratio'],
                'Max DD': metrics['max_drawdown'],
                'Calmar': metrics['calmar_ratio'],
                'Win Rate': metrics['win_rate']
            })
        
        return pd.DataFrame(comparison)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download data
    data = yf.download('ICLN', start='2020-01-01', end='2024-12-31')
    prices = pd.DataFrame({'ICLN': data['Close']})
    
    # Create dummy positions (simple momentum)
    returns = prices.pct_change()
    momentum = returns.rolling(20).mean()
    positions = pd.DataFrame(np.where(momentum > 0, 0.30, -0.10), 
                            index=prices.index, 
                            columns=prices.columns)
    
    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage_bps=2.0
    )
    
    engine = ProfessionalBacktestEngine(config)
    results = engine.run_backtest(positions, prices)
