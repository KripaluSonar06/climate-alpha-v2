"""
PROFESSIONAL PORTFOLIO OPTIMIZER
Kelly Criterion + Risk Parity + Black-Litterman
Multi-asset optimization for hedge-fund-grade portfolio construction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class ProfessionalPortfolioOptimizer:
    """
    Institutional-grade portfolio optimizer
    Methods: Kelly Criterion, Risk Parity, Mean-Variance, Black-Litterman
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize optimizer
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.max_leverage = 2.0
        self.max_position = 0.40  # Max 40% in single asset
        self.min_position = -0.20  # Allow some short positions
        
        print("Professional Portfolio Optimizer initialized")
    
    def kelly_criterion_position_size(self, 
                                     win_rate: float,
                                     avg_win: float,
                                     avg_loss: float,
                                     kelly_fraction: float = 0.25) -> float:
        """
        Calculate Kelly Criterion position size
        
        Args:
            win_rate: Probability of winning trade
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            kelly_fraction: Fraction of Kelly to use (default: 0.25 = quarter Kelly)
            
        Returns:
            Optimal position size
        """
        if avg_loss == 0 or win_rate == 0:
            return 0.0
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win rate, q = 1-p, b = avg_win/avg_loss
        b = avg_win / avg_loss
        q = 1 - win_rate
        
        kelly = (win_rate * b - q) / b
        
        # Apply Kelly fraction (reduce to be more conservative)
        position_size = max(0, kelly * kelly_fraction)
        
        # Cap at max position
        return min(position_size, self.max_position)
    
    def calculate_kelly_from_predictions(self,
                                        predictions: pd.Series,
                                        actual_returns: pd.Series) -> pd.Series:
        """
        Calculate Kelly position sizes from ML predictions
        
        Args:
            predictions: Predicted returns
            actual_returns: Actual historical returns (for calibration)
            
        Returns:
            Kelly position sizes
        """
        # Align predictions with actual returns
        aligned_pred = predictions.reindex(actual_returns.index).dropna()
        aligned_actual = actual_returns.reindex(aligned_pred.index)
        
        # Calculate win rate and avg win/loss
        correct_direction = np.sign(aligned_pred) == np.sign(aligned_actual)
        win_rate = correct_direction.sum() / len(correct_direction)
        
        wins = aligned_actual[aligned_actual > 0]
        losses = abs(aligned_actual[aligned_actual < 0])
        
        avg_win = wins.mean() if len(wins) > 0 else 0.01
        avg_loss = losses.mean() if len(losses) > 0 else 0.01
        
        # Calculate Kelly for each prediction
        kelly_positions = pd.Series(index=predictions.index, dtype=float)
        
        for idx in predictions.index:
            pred = predictions.loc[idx]
            
            # Use prediction confidence as win probability adjustment
            confidence = min(abs(pred) * 10, 0.9)  # Scale prediction to probability
            adjusted_win_rate = win_rate * confidence + 0.5 * (1 - confidence)
            
            kelly_size = self.kelly_criterion_position_size(
                adjusted_win_rate, avg_win, avg_loss, kelly_fraction=0.25
            )
            
            # Apply direction from prediction
            kelly_positions.loc[idx] = kelly_size * np.sign(pred)
        
        return kelly_positions
    
    def risk_parity_weights(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate risk parity weights
        Each asset contributes equally to portfolio risk
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Risk parity weights
        """
        # Calculate volatilities
        volatilities = returns.std()
        
        # Inverse volatility weights
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        return weights
    
    def mean_variance_optimization(self,
                                  expected_returns: pd.Series,
                                  covariance_matrix: pd.DataFrame,
                                  target_return: Optional[float] = None) -> pd.Series:
        """
        Mean-variance optimization (Markowitz)
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            target_return: Target portfolio return (if None, maximize Sharpe)
            
        Returns:
            Optimal weights
        """
        n_assets = len(expected_returns)
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Bounds: min_position to max_position for each asset
        bounds = tuple((self.min_position, self.max_position) for _ in range(n_assets))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: expected_returns @ x - target_return
            })
        
        # Objective: minimize variance (or maximize Sharpe if no target return)
        if target_return is None:
            # Maximize Sharpe ratio
            def neg_sharpe(weights):
                port_return = expected_returns @ weights
                port_vol = np.sqrt(weights @ covariance_matrix @ weights)
                return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 999
            
            result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            # Minimize variance
            def portfolio_variance(weights):
                return weights @ covariance_matrix @ weights
            
            result = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return pd.Series(result.x, index=expected_returns.index)
        else:
            # Fallback to equal weights
            print("Optimization failed, using equal weights")
            return pd.Series(1/n_assets, index=expected_returns.index)
    
    def black_litterman_optimization(self,
                                    market_weights: pd.Series,
                                    views: Dict[str, float],
                                    view_confidences: Dict[str, float],
                                    returns: pd.DataFrame) -> pd.Series:
        """
        Black-Litterman model - combines market equilibrium with investor views
        
        Args:
            market_weights: Current market capitalization weights
            views: Dictionary of {asset: expected_return}
            view_confidences: Dictionary of {asset: confidence_level} (0-1)
            returns: Historical returns for covariance
            
        Returns:
            Optimal weights incorporating views
        """
        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252
        
        # Market implied returns (reverse optimization)
        risk_aversion = 2.5
        market_returns = risk_aversion * cov_matrix @ market_weights
        
        # Build view matrix P and view returns Q
        n_assets = len(returns.columns)
        n_views = len(views)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        omega = np.zeros((n_views, n_views))
        
        asset_list = list(returns.columns)
        
        for i, (asset, view_return) in enumerate(views.items()):
            if asset in asset_list:
                asset_idx = asset_list.index(asset)
                P[i, asset_idx] = 1
                Q[i] = view_return
                
                # Confidence -> omega (uncertainty in views)
                confidence = view_confidences.get(asset, 0.5)
                omega[i, i] = (1 - confidence) * cov_matrix.iloc[asset_idx, asset_idx]
        
        # Black-Litterman formula
        tau = 0.025  # Scaling factor
        
        # Posterior returns
        inv_cov = np.linalg.inv(cov_matrix)
        inv_omega = np.linalg.inv(omega) if np.linalg.det(omega) != 0 else np.eye(n_views)
        
        posterior_cov_inv = inv_cov + P.T @ inv_omega @ P / tau
        posterior_cov = np.linalg.inv(posterior_cov_inv)
        
        posterior_returns = posterior_cov @ (
            inv_cov @ market_returns + P.T @ inv_omega @ Q / tau
        )
        
        # Optimize with posterior returns
        optimal_weights = self.mean_variance_optimization(
            pd.Series(posterior_returns, index=returns.columns),
            pd.DataFrame(cov_matrix, index=returns.columns, columns=returns.columns)
        )
        
        return optimal_weights
    
    def dynamic_position_sizing(self,
                               signal: float,
                               current_volatility: float,
                               target_volatility: float = 0.15,
                               base_size: float = 0.30) -> float:
        """
        Dynamic position sizing based on volatility targeting
        
        Args:
            signal: Trading signal (-1 to 1)
            current_volatility: Current asset volatility (annualized)
            target_volatility: Target portfolio volatility (default: 15%)
            base_size: Base position size (default: 30%)
            
        Returns:
            Adjusted position size
        """
        # Volatility scaling
        vol_scalar = target_volatility / current_volatility if current_volatility > 0 else 1.0
        vol_scalar = np.clip(vol_scalar, 0.5, 2.0)  # Limit scaling
        
        # Apply to base size
        adjusted_size = base_size * vol_scalar * signal
        
        # Clip to position limits
        return np.clip(adjusted_size, self.min_position, self.max_position)
    
    def multi_asset_portfolio(self,
                             predictions: Dict[str, pd.Series],
                             returns: pd.DataFrame,
                             method: str = 'kelly') -> pd.DataFrame:
        """
        Construct multi-asset portfolio using specified method
        
        Args:
            predictions: Dictionary of {asset: prediction_series}
            returns: DataFrame of historical returns
            method: Optimization method ('kelly', 'risk_parity', 'mean_variance')
            
        Returns:
            DataFrame of position sizes over time
        """
        # Align all predictions
        common_dates = returns.index
        for asset, pred in predictions.items():
            common_dates = common_dates.intersection(pred.index)
        
        positions = pd.DataFrame(index=common_dates, columns=returns.columns)
        
        if method == 'kelly':
            # Kelly criterion for each asset
            for asset in returns.columns:
                if asset in predictions:
                    kelly_pos = self.calculate_kelly_from_predictions(
                        predictions[asset],
                        returns[asset]
                    )
                    positions[asset] = kelly_pos.reindex(common_dates)
        
        elif method == 'risk_parity':
            # Risk parity weights applied to combined signal
            weights = self.risk_parity_weights(returns)
            
            for asset in returns.columns:
                if asset in predictions:
                    signal = predictions[asset].reindex(common_dates)
                    positions[asset] = signal * weights[asset]
        
        elif method == 'mean_variance':
            # Rolling mean-variance optimization
            window = 60
            
            for i in range(window, len(common_dates)):
                date = common_dates[i]
                hist_returns = returns.loc[common_dates[i-window:i]]
                
                # Expected returns from predictions
                expected_returns = pd.Series({
                    asset: predictions[asset].loc[date] if asset in predictions else 0
                    for asset in returns.columns
                })
                
                # Optimize
                cov_matrix = hist_returns.cov() * 252
                weights = self.mean_variance_optimization(expected_returns, cov_matrix)
                
                positions.loc[date] = weights
        
        # Ensure no NaN and within limits
        positions = positions.fillna(0)
        positions = positions.clip(self.min_position, self.max_position)
        
        return positions


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download data
    tickers = ['ICLN', 'TAN', 'ENPH']
    data = yf.download(tickers, start='2020-01-01', end='2024-12-31')
    returns = data['Close'].pct_change().dropna()
    
    # Initialize optimizer
    optimizer = ProfessionalPortfolioOptimizer()
    
    # Test Kelly criterion
    kelly_size = optimizer.kelly_criterion_position_size(
        win_rate=0.55,
        avg_win=0.02,
        avg_loss=0.015,
        kelly_fraction=0.25
    )
    print(f"Kelly position size: {kelly_size:.2%}")
    
    # Test risk parity
    rp_weights = optimizer.risk_parity_weights(returns)
    print(f"\nRisk Parity Weights:")
    print(rp_weights)
