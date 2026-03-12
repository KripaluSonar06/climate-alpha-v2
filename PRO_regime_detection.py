"""
PROFESSIONAL REGIME DETECTION SYSTEM
Identifies market regimes using multiple methods:
- Hidden Markov Models (HMM)
- Statistical clustering
- Volatility-based classification
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.cluster import KMeans
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class RegimeDetector:
    """
    Professional regime detection for adaptive strategy management
    Identifies: Bull, Bear, High Volatility, Low Volatility, Trending, Range-bound
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector
        
        Args:
            n_regimes: Number of regimes to detect (default: 3 = Bull/Bear/Neutral)
        """
        self.n_regimes = n_regimes
        self.is_fitted = False
        self.regime_names = {
            3: ['Bear', 'Neutral', 'Bull'],
            4: ['Bear', 'Low Vol', 'High Vol', 'Bull'],
            5: ['Crash', 'Bear', 'Neutral', 'Bull', 'Boom']
        }
        
        print(f"Initializing Regime Detector with {n_regimes} regimes")
        
    def detect_regimes_hmm(self, returns: pd.Series, 
                          volatility: pd.Series = None) -> pd.Series:
        """
        Detect regimes using Hidden Markov Model
        
        Args:
            returns: Return series
            volatility: Volatility series (optional)
            
        Returns:
            Series of regime labels
        """
        try:
            from hmmlearn import hmm
        except ImportError:
            print("Warning: hmmlearn not installed. Using fallback method.")
            return self._detect_regimes_statistical(returns, volatility)
        
        # Prepare features
        if volatility is not None:
            features = np.column_stack([returns.values, volatility.values])
        else:
            vol = returns.rolling(20).std()
            features = np.column_stack([returns.values, vol.values])
        
        # Remove NaN
        mask = ~np.isnan(features).any(axis=1)
        features_clean = features[mask]
        
        # Fit HMM
        model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=100,
            random_state=42
        )
        
        model.fit(features_clean)
        
        # Predict regimes
        regimes = np.full(len(features), -1)
        regimes[mask] = model.predict(features_clean)
        
        # Map to meaningful labels based on mean returns
        regime_returns = {}
        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            if regime_mask.sum() > 0:
                regime_returns[regime] = returns[regime_mask].mean()
        
        # Sort regimes by returns (low to high)
        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        
        # Remap
        regimes_mapped = np.array([regime_mapping.get(r, r) for r in regimes])
        
        return pd.Series(regimes_mapped, index=returns.index)
    
    def _detect_regimes_statistical(self, returns: pd.Series,
                                   volatility: pd.Series = None) -> pd.Series:
        """
        Fallback: Statistical regime detection using K-means
        
        Args:
            returns: Return series
            volatility: Volatility series
            
        Returns:
            Series of regime labels
        """
        # Calculate features
        ma_short = returns.rolling(20).mean()
        ma_long = returns.rolling(60).mean()
        
        if volatility is None:
            volatility = returns.rolling(20).std()
        
        # Prepare features
        features = pd.DataFrame({
            'returns': returns,
            'ma_short': ma_short,
            'ma_long': ma_long,
            'volatility': volatility,
            'trend': ma_short - ma_long
        }).dropna()
        
        # K-means clustering
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        regimes = kmeans.fit_predict(features[['returns', 'volatility', 'trend']])
        
        # Map to series
        regime_series = pd.Series(-1, index=returns.index)
        regime_series.loc[features.index] = regimes
        
        # Sort by mean return
        regime_returns = {}
        for regime in range(self.n_regimes):
            mask = regime_series == regime
            if mask.sum() > 0:
                regime_returns[regime] = returns[mask].mean()
        
        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        
        regime_series = regime_series.map(lambda x: regime_mapping.get(x, x))
        
        return regime_series
    
    def detect_regimes_comprehensive(self, prices: pd.Series) -> pd.DataFrame:
        """
        Comprehensive regime detection using multiple methods
        
        Args:
            prices: Price series
            
        Returns:
            DataFrame with multiple regime classifications
        """
        returns = prices.pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        
        regimes = pd.DataFrame(index=prices.index)
        
        # 1. Statistical regimes (using K-means)
        regimes['regime_statistical'] = self._detect_regimes_statistical(returns, volatility)
        
        # 2. Volatility-based regimes
        regimes['regime_volatility'] = self._volatility_regimes(volatility)
        
        # 3. Trend-based regimes
        regimes['regime_trend'] = self._trend_regimes(prices)
        
        # 4. Combined regime (majority vote)
        regimes['regime_combined'] = regimes[['regime_statistical', 
                                              'regime_volatility', 
                                              'regime_trend']].mode(axis=1)[0]
        
        # 5. Regime probabilities
        regimes = self._calculate_regime_probabilities(regimes, returns, volatility)
        
        return regimes
    
    def _volatility_regimes(self, volatility: pd.Series) -> pd.Series:
        """Classify regimes based on volatility quantiles"""
        
        vol_quantiles = volatility.quantile([0.33, 0.67])
        
        regimes = pd.Series(1, index=volatility.index)  # Normal vol
        regimes[volatility < vol_quantiles.iloc[0]] = 0  # Low vol
        regimes[volatility > vol_quantiles.iloc[1]] = 2  # High vol
        
        return regimes
    
    def _trend_regimes(self, prices: pd.Series) -> pd.Series:
        """Classify regimes based on trend"""
        
        ma_fast = prices.rolling(20).mean()
        ma_slow = prices.rolling(60).mean()
        
        trend = (ma_fast - ma_slow) / ma_slow
        
        regimes = pd.Series(1, index=prices.index)  # Neutral
        regimes[trend < -0.02] = 0  # Bear
        regimes[trend > 0.02] = 2   # Bull
        
        return regimes
    
    def _calculate_regime_probabilities(self, regimes: pd.DataFrame,
                                       returns: pd.Series,
                                       volatility: pd.Series) -> pd.DataFrame:
        """
        Calculate probability of being in each regime
        
        Args:
            regimes: Regime classifications
            returns: Return series
            volatility: Volatility series
            
        Returns:
            DataFrame with regime probabilities added
        """
        # Use regime_combined for probability calculation
        regime_col = 'regime_combined'
        
        for r in range(self.n_regimes):
            # Calculate parameters for this regime
            mask = regimes[regime_col] == r
            if mask.sum() > 10:
                mu = returns[mask].mean()
                sigma = returns[mask].std()
                
                # Likelihood of current observation given regime
                likelihood = norm.pdf(returns, mu, sigma)
                regimes[f'prob_regime_{r}'] = likelihood
            else:
                regimes[f'prob_regime_{r}'] = 0
        
        # Normalize probabilities
        prob_cols = [c for c in regimes.columns if c.startswith('prob_regime_')]
        prob_sum = regimes[prob_cols].sum(axis=1)
        for col in prob_cols:
            regimes[col] = regimes[col] / (prob_sum + 1e-10)
        
        return regimes
    
    def get_regime_statistics(self, regimes: pd.Series, 
                             returns: pd.Series) -> pd.DataFrame:
        """
        Calculate statistics for each regime
        
        Args:
            regimes: Regime labels
            returns: Return series
            
        Returns:
            DataFrame with regime statistics
        """
        stats = []
        
        regime_names = self.regime_names.get(self.n_regimes, 
                                             [f'Regime_{i}' for i in range(self.n_regimes)])
        
        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) > 0:
                stats.append({
                    'Regime': regime_names[regime],
                    'Occurrences': mask.sum(),
                    'Percentage': mask.sum() / len(regimes) * 100,
                    'Mean Return': regime_returns.mean() * 252,
                    'Volatility': regime_returns.std() * np.sqrt(252),
                    'Sharpe': (regime_returns.mean() / regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0,
                    'Max Return': regime_returns.max(),
                    'Min Return': regime_returns.min()
                })
        
        return pd.DataFrame(stats)
    
    def get_regime_transitions(self, regimes: pd.Series) -> pd.DataFrame:
        """
        Calculate regime transition matrix
        
        Args:
            regimes: Regime labels
            
        Returns:
            Transition probability matrix
        """
        # Create transition matrix
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(len(regimes) - 1):
            if regimes.iloc[i] >= 0 and regimes.iloc[i+1] >= 0:
                from_regime = int(regimes.iloc[i])
                to_regime = int(regimes.iloc[i+1])
                transition_matrix[from_regime, to_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                     where=row_sums!=0, 
                                     out=np.zeros_like(transition_matrix))
        
        regime_names = self.regime_names.get(self.n_regimes,
                                             [f'Regime_{i}' for i in range(self.n_regimes)])
        
        return pd.DataFrame(transition_matrix, 
                          index=regime_names, 
                          columns=regime_names)


class AdaptiveStrategyManager:
    """
    Manages strategy parameters based on detected regimes
    """
    
    def __init__(self):
        self.regime_params = {
            0: {'position_multiplier': 0.5, 'stop_loss': 0.02, 'name': 'Bear'},
            1: {'position_multiplier': 1.0, 'stop_loss': 0.03, 'name': 'Neutral'},
            2: {'position_multiplier': 1.5, 'stop_loss': 0.04, 'name': 'Bull'}
        }
    
    def get_position_adjustment(self, regime: int) -> float:
        """Get position size multiplier for current regime"""
        return self.regime_params.get(regime, {}).get('position_multiplier', 1.0)
    
    def get_stop_loss(self, regime: int) -> float:
        """Get stop loss threshold for current regime"""
        return self.regime_params.get(regime, {}).get('stop_loss', 0.03)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download data
    data = yf.download('ICLN', start='2019-01-01', end='2024-12-31')
    prices = data['Close']
    returns = prices.pct_change()
    
    # Detect regimes
    detector = RegimeDetector(n_regimes=3)
    regimes_comprehensive = detector.detect_regimes_comprehensive(prices)
    
    print("\nRegime Detection Complete!")
    print(f"Regime columns: {list(regimes_comprehensive.columns)}")
    
    # Get statistics
    stats = detector.get_regime_statistics(
        regimes_comprehensive['regime_combined'], 
        returns
    )
    print("\nRegime Statistics:")
    print(stats.to_string())
    
    # Get transitions
    transitions = detector.get_regime_transitions(regimes_comprehensive['regime_combined'])
    print("\nRegime Transition Matrix:")
    print(transitions.to_string())
