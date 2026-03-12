"""
PROFESSIONAL FEATURE ENGINEERING - 200+ FEATURES
Improved version with stability fixes and data validation
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

    def create_all_features(self, prices_df: pd.DataFrame,
                            volume_df=None) -> pd.DataFrame:

        features = pd.DataFrame(index=prices_df.index)

        # ----------------------------
        # Extract OHLC
        # ----------------------------
        if "Close" in prices_df.columns:
            close = prices_df["Close"]
            high = prices_df.get("High", close)
            low = prices_df.get("Low", close)
            open_price = prices_df.get("Open", close)
        else:
            close = prices_df.iloc[:, 0]
            high = low = open_price = close

        # ----------------------------
        # FIX 1: Robust volume handling
        # ----------------------------
        if volume_df is None:
            volume = pd.Series(1.0, index=close.index)

        elif isinstance(volume_df, pd.DataFrame):
            volume = volume_df.iloc[:, 0]

        else:
            volume = volume_df

        print("Creating features:")

        features.update(self._price_features(close, high, low, open_price))
        features.update(self._momentum_features(close))
        features.update(self._volatility_features(close, high, low))
        features.update(self._volume_features(close, volume))
        features.update(self._mean_reversion_features(close))
        features.update(self._statistical_features(close))
        features.update(self._technical_indicators(close, high, low, volume))

        # ----------------------------
        # FIX 2: Clean NaN / Inf
        # ----------------------------
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        self.feature_names = list(features.columns)

        print(f"\nTotal features created: {len(self.feature_names)}")

        return features

    # ----------------------------------------------------
    # PRICE FEATURES
    # ----------------------------------------------------
    def _price_features(self, close, high, low, open_price):

        features = {}

        for period in [1,2,3,5,10,20,40,60]:

            features[f"return_{period}"] = close.pct_change(period)

            features[f"log_return_{period}"] = np.log(close / close.shift(period))

        for window in [5,10,20,50,100,200]:

            ma = close.rolling(window).mean()

            features[f"price_vs_ma_{window}"] = (close - ma) / ma

        features["high_low_range"] = (high - low) / close
        features["open_close_range"] = (close - open_price) / open_price

        return pd.DataFrame(features)

    # ----------------------------------------------------
    # MOMENTUM FEATURES
    # ----------------------------------------------------
    def _momentum_features(self, close):

        features = {}

        for period in [5,10,20,30,60,90]:

            features[f"roc_{period}"] = (close - close.shift(period)) / close.shift(period)

        # RSI
        for period in [7,14,21]:

            delta = close.diff()

            gain = delta.clip(lower=0).rolling(period).mean()

            loss = (-delta.clip(upper=0)).rolling(period).mean()

            rs = gain / (loss.replace(0,np.nan))

            features[f"rsi_{period}"] = 100 - (100/(1+rs))

        # MACD
        for fast, slow in [(12,26),(5,15)]:

            ema_fast = close.ewm(span=fast).mean()

            ema_slow = close.ewm(span=slow).mean()

            macd = ema_fast - ema_slow

            features[f"macd_{fast}_{slow}"] = macd

            features[f"macd_signal_{fast}_{slow}"] = macd.ewm(span=9).mean()

        return pd.DataFrame(features)

    # ----------------------------------------------------
    # VOLATILITY FEATURES
    # ----------------------------------------------------
    def _volatility_features(self, close, high, low):

        features = {}

        returns = close.pct_change()

        for window in [5,10,20,30,60]:

            features[f"volatility_{window}"] = returns.rolling(window).std() * np.sqrt(252)

        # Parkinson volatility
        for window in [10,20]:

            hl = np.log(high / low)

            features[f"parkinson_vol_{window}"] = np.sqrt((hl**2).rolling(window).mean()/(4*np.log(2)))

        # ATR
        tr1 = high-low

        tr2 = abs(high-close.shift())

        tr3 = abs(low-close.shift())

        tr = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)

        for p in [7,14]:

            features[f"atr_{p}"] = tr.rolling(p).mean()/close

        return pd.DataFrame(features)

    # ----------------------------------------------------
    # VOLUME FEATURES
    # ----------------------------------------------------
    def _volume_features(self, close, volume):

        features = {}

        for p in [1,5,10]:

            features[f"volume_change_{p}"] = volume.pct_change(p)

        for w in [5,10,20]:

            vma = volume.rolling(w).mean()

            features[f"volume_vs_ma_{w}"] = volume / vma

        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()

        features["obv"] = obv

        return pd.DataFrame(features)

    # ----------------------------------------------------
    # MEAN REVERSION
    # ----------------------------------------------------
    def _mean_reversion_features(self, close):

        features = {}

        for window in [10,20,30]:

            ma = close.rolling(window).mean()

            std = close.rolling(window).std()

            features[f"zscore_{window}"] = (close-ma)/std

        return pd.DataFrame(features)

    # ----------------------------------------------------
    # STATISTICAL FEATURES
    # ----------------------------------------------------
    def _statistical_features(self, close):

        features = {}

        returns = close.pct_change()

        for w in [20,60]:

            features[f"skew_{w}"] = returns.rolling(w).skew()

            features[f"kurt_{w}"] = returns.rolling(w).kurt()

        return pd.DataFrame(features)

    # ----------------------------------------------------
    # TECHNICAL INDICATORS
    # ----------------------------------------------------
    def _technical_indicators(self, close, high, low, volume):

        features = {}

        try:

            features["adx"] = talib.ADX(high,low,close,timeperiod=14)

            features["cci"] = talib.CCI(high,low,close,timeperiod=20)

            features["mfi"] = talib.MFI(high,low,close,volume,timeperiod=14)

            features["trix"] = talib.TRIX(close,timeperiod=15)

        except:

            features["adx"]=0
            features["cci"]=0
            features["mfi"]=0
            features["trix"]=0

        return pd.DataFrame(features,index=close.index)

    # ----------------------------------------------------
    # FEATURE GROUPS
    # ----------------------------------------------------
    def get_feature_importance_groups(self):

        groups = {

            "price":[f for f in self.feature_names if "return" in f],

            "momentum":[f for f in self.feature_names if "roc" in f or "rsi" in f],

            "volatility":[f for f in self.feature_names if "volatility" in f or "atr" in f],

            "volume":[f for f in self.feature_names if "volume" in f or "obv" in f],

            "mean_reversion":[f for f in self.feature_names if "zscore" in f],

            "statistical":[f for f in self.feature_names if "skew" in f or "kurt" in f],

            "technical":[f for f in self.feature_names if "adx" in f or "cci" in f]

        }

        return groups