"""
PROFESSIONAL FEATURE ENGINEERING
Stable version (no scalar-index errors)
"""

import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


class AdvancedFeatureEngineer:

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []

    # ----------------------------------------------------
    # MAIN FEATURE CREATOR
    # ----------------------------------------------------

    def create_all_features(self, prices_df: pd.DataFrame, volume_df=None) -> pd.DataFrame:

        # Ensure dataframe
        prices_df = prices_df.copy()

        # Force numeric
        prices_df = prices_df.apply(pd.to_numeric, errors="coerce")

        # Extract OHLC safely
        close = prices_df["Close"] if "Close" in prices_df.columns else prices_df.iloc[:, 0]
        high = prices_df["High"] if "High" in prices_df.columns else close
        low = prices_df["Low"] if "Low" in prices_df.columns else close
        open_price = prices_df["Open"] if "Open" in prices_df.columns else close

        # Force Series
        close = pd.Series(close)
        high = pd.Series(high)
        low = pd.Series(low)
        open_price = pd.Series(open_price)

        # Volume handling
        if volume_df is None:
            volume = pd.Series(1.0, index=close.index)

        elif isinstance(volume_df, pd.DataFrame):
            volume = volume_df.iloc[:, 0]

        else:
            volume = volume_df

        volume = pd.Series(volume, index=close.index)

        print("Creating features:")

        features = pd.DataFrame(index=close.index)

        features = pd.concat([
            self._price_features(close, high, low, open_price),
            self._momentum_features(close),
            self._volatility_features(close, high, low),
            self._volume_features(close, volume),
            self._mean_reversion_features(close),
            self._statistical_features(close),
            self._technical_indicators(close, high, low, volume)
        ], axis=1)

        # Clean data
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        self.feature_names = list(features.columns)

        print(f"Total features created: {len(self.feature_names)}")

        return features

    # ----------------------------------------------------
    # PRICE FEATURES
    # ----------------------------------------------------

    def _price_features(self, close, high, low, open_price):

        df = pd.DataFrame(index=close.index)

        for period in [1,2,3,5,10,20,40,60]:

            df[f"return_{period}"] = close.pct_change(period)
            df[f"log_return_{period}"] = np.log(close / close.shift(period))

        for window in [5,10,20,50,100,200]:

            ma = close.rolling(window).mean()
            df[f"price_vs_ma_{window}"] = (close - ma) / ma

        df["high_low_range"] = (high - low) / close
        df["open_close_range"] = (close - open_price) / open_price

        return df

    # ----------------------------------------------------
    # MOMENTUM FEATURES
    # ----------------------------------------------------

    def _momentum_features(self, close):

        df = pd.DataFrame(index=close.index)

        for period in [5,10,20,30,60,90]:

            df[f"roc_{period}"] = (close - close.shift(period)) / close.shift(period)

        # RSI
        for period in [7,14,21]:

            delta = close.diff()

            gain = delta.clip(lower=0).rolling(period).mean()
            loss = (-delta.clip(upper=0)).rolling(period).mean()

            rs = gain / (loss.replace(0, np.nan))

            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # MACD
        for fast, slow in [(12,26),(5,15)]:

            ema_fast = close.ewm(span=fast).mean()
            ema_slow = close.ewm(span=slow).mean()

            macd = ema_fast - ema_slow

            df[f"macd_{fast}_{slow}"] = macd
            df[f"macd_signal_{fast}_{slow}"] = macd.ewm(span=9).mean()

        return df

    # ----------------------------------------------------
    # VOLATILITY FEATURES
    # ----------------------------------------------------

    def _volatility_features(self, close, high, low):

        df = pd.DataFrame(index=close.index)

        returns = close.pct_change()

        for window in [5,10,20,30,60]:

            df[f"volatility_{window}"] = returns.rolling(window).std() * np.sqrt(252)

        # Parkinson volatility
        for window in [10,20]:

            hl = np.log(high / low)

            df[f"parkinson_vol_{window}"] = np.sqrt((hl**2).rolling(window).mean()/(4*np.log(2)))

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        for p in [7,14]:

            df[f"atr_{p}"] = tr.rolling(p).mean() / close

        return df

    # ----------------------------------------------------
    # VOLUME FEATURES
    # ----------------------------------------------------

    def _volume_features(self, close, volume):

        df = pd.DataFrame(index=close.index)

        for p in [1,5,10]:

            df[f"volume_change_{p}"] = volume.pct_change(p)

        for w in [5,10,20]:

            vma = volume.rolling(w).mean()
            df[f"volume_vs_ma_{w}"] = volume / vma

        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()

        df["obv"] = obv

        return df

    # ----------------------------------------------------
    # MEAN REVERSION FEATURES
    # ----------------------------------------------------

    def _mean_reversion_features(self, close):

        df = pd.DataFrame(index=close.index)

        for window in [10,20,30]:

            ma = close.rolling(window).mean()
            std = close.rolling(window).std()

            df[f"zscore_{window}"] = (close - ma) / std

        return df

    # ----------------------------------------------------
    # STATISTICAL FEATURES
    # ----------------------------------------------------

    def _statistical_features(self, close):

        df = pd.DataFrame(index=close.index)

        returns = close.pct_change()

        for w in [20,60]:

            df[f"skew_{w}"] = returns.rolling(w).skew()
            df[f"kurt_{w}"] = returns.rolling(w).kurt()

        return df

    # ----------------------------------------------------
    # TECHNICAL INDICATORS
    # ----------------------------------------------------

    def _technical_indicators(self, close, high, low, volume):

        df = pd.DataFrame(index=close.index)

        try:

            df["adx"] = talib.ADX(high, low, close, timeperiod=14)
            df["cci"] = talib.CCI(high, low, close, timeperiod=20)
            df["mfi"] = talib.MFI(high, low, close, volume, timeperiod=14)
            df["trix"] = talib.TRIX(close, timeperiod=15)

        except:

            df["adx"] = 0
            df["cci"] = 0
            df["mfi"] = 0
            df["trix"] = 0

        return df

    # ----------------------------------------------------
    # FEATURE GROUPS
    # ----------------------------------------------------

    def get_feature_importance_groups(self):

        groups = {

            "price": [f for f in self.feature_names if "return" in f],
            "momentum": [f for f in self.feature_names if "roc" in f or "rsi" in f],
            "volatility": [f for f in self.feature_names if "volatility" in f or "atr" in f],
            "volume": [f for f in self.feature_names if "volume" in f or "obv" in f],
            "mean_reversion": [f for f in self.feature_names if "zscore" in f],
            "statistical": [f for f in self.feature_names if "skew" in f or "kurt" in f],
            "technical": [f for f in self.feature_names if "adx" in f or "cci" in f]

        }

        return groups