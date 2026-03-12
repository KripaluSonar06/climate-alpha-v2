"""
CLIMATE-ALPHA v4 MASTER STRATEGY
Institutional Quant Research Pipeline
"""

import numpy as np
import pandas as pd
import yfinance as yf

from typing import List
from concurrent.futures import ThreadPoolExecutor

from sklearn.feature_selection import SelectKBest, f_regression

from PRO_feature_engineering import AdvancedFeatureEngineer
from PRO_ensemble_models import EnsembleMLSystem
from PRO_regime_detection import RegimeDetector
from PRO_portfolio_optimizer import ProfessionalPortfolioOptimizer
from PRO_backtest_engine import ProfessionalBacktestEngine, BacktestConfig


class ClimateAlphaMasterStrategy:

    def __init__(self, tickers: List[str], start_date, end_date):

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        self.feature_engineer = AdvancedFeatureEngineer()
        self.regime_detector = RegimeDetector(n_regimes=3)
        self.portfolio_optimizer = ProfessionalPortfolioOptimizer()

        self.backtest_engine = ProfessionalBacktestEngine(
            BacktestConfig(
                initial_capital=100000,
                transaction_cost=0.0015,
                slippage_bps=3,
                max_leverage=1.5,
                max_position_size=0.40
            )
        )

        self.data = {}
        self.features = {}
        self.predictions = {}
        self.models = {}

        self.positions = None
        self.regimes = None
        self.results = None

        print("\nCLIMATE ALPHA v4 INITIALIZED")

    # ------------------------------------------------
    # DATA COLLECTION
    # ------------------------------------------------

    def step1_collect_data(self):

        print("\nSTEP 1: DATA DOWNLOAD")

        for ticker in self.tickers:

            df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False
            )

            df = df.dropna()

            self.data[ticker] = df

            print(f"{ticker}: {len(df)} rows")

    # ------------------------------------------------
    # FEATURE ENGINEERING
    # ------------------------------------------------

    def step2_features(self):

        print("\nSTEP 2: FEATURE ENGINEERING")

        for ticker in self.tickers:

            df = self.data[ticker]

            feats = self.feature_engineer.create_all_features(
                df,
                df["Volume"] if "Volume" in df.columns else None
            )

            feats = self._feature_selection(feats, df["Close"])

            self.features[ticker] = feats

            print(f"{ticker}: {feats.shape[1]} features")

    # ------------------------------------------------
    # FEATURE SELECTION
    # ------------------------------------------------

    def _feature_selection(self, features, price):

        target = price.pct_change().shift(-5)

        df = features.copy()
        df["target"] = target

        df = df.dropna()

        X = df.drop("target", axis=1)
        y = df["target"]

        selector = SelectKBest(f_regression, k=60)
        selector.fit(X, y)

        selected = X.columns[selector.get_support()]

        return features[selected]

    # ------------------------------------------------
    # ML TRAINING
    # ------------------------------------------------

    def step3_ml_training(self):

        print("\nSTEP 3: ML TRAINING")

        def train_asset(ticker):

            feats = self.features[ticker]
            prices = self.data[ticker]["Close"]

            model = EnsembleMLSystem(target_horizon=5)

            model.train_ensemble(feats, prices)

            preds = model.predict(feats)

            return ticker, model, preds

        with ThreadPoolExecutor(max_workers=len(self.tickers)) as executor:

            results = executor.map(train_asset, self.tickers)

        for ticker, model, preds in results:

            self.models[ticker] = model
            self.predictions[ticker] = preds

            print(f"{ticker}: model trained")

    # ------------------------------------------------
    # REGIME DETECTION
    # ------------------------------------------------

    def step4_regimes(self):

        print("\nSTEP 4: REGIME DETECTION")

        avg_price = pd.concat(
            [self.data[t]["Close"] for t in self.tickers],
            axis=1
        ).mean(axis=1)

        self.regimes = self.regime_detector.detect_regimes_comprehensive(avg_price)

        print("Market regimes identified")

    # ------------------------------------------------
    # PORTFOLIO CONSTRUCTION
    # ------------------------------------------------

    def step5_portfolio(self):

        print("\nSTEP 5: PORTFOLIO CONSTRUCTION")

        returns = pd.DataFrame({
            t: self.data[t]["Close"].pct_change()
            for t in self.tickers
        })

        positions = {}

        for ticker in self.tickers:

            pos = self.portfolio_optimizer.calculate_kelly_from_predictions(
                self.predictions[ticker],
                returns[ticker]
            )

            positions[ticker] = pos

        self.positions = pd.DataFrame(positions)

        self._apply_regime_scaling()
        self._apply_volatility_target()
        self._correlation_filter()

    # ------------------------------------------------

    def _apply_regime_scaling(self):

        if self.regimes is None:
            return

        regime = self.regimes["regime_combined"]

        multipliers = {0: 0.5, 1: 1.0, 2: 1.3}

        for r, m in multipliers.items():

            mask = regime == r
            self.positions.loc[mask] *= m

    # ------------------------------------------------

    def _apply_volatility_target(self):

        returns = pd.DataFrame({
            t: self.data[t]["Close"].pct_change()
            for t in self.tickers
        })

        vol = returns.rolling(20).std()

        target_vol = 0.15

        scaler = target_vol / (vol + 1e-6)

        scaler = scaler.clip(0.5, 2)

        self.positions *= scaler

    # ------------------------------------------------

    def _correlation_filter(self):

        returns = pd.DataFrame({
            t: self.data[t]["Close"].pct_change()
            for t in self.tickers
        })

        corr = returns.corr()

        for i in range(len(corr)):
            for j in range(i+1, len(corr)):

                if abs(corr.iloc[i, j]) > 0.8:

                    asset = corr.columns[j]

                    self.positions[asset] *= 0.5

    # ------------------------------------------------
    # BACKTEST
    # ------------------------------------------------

    def step6_backtest(self):

        print("\nSTEP 6: BACKTEST")

        prices = pd.DataFrame({
            t: self.data[t]["Close"]
            for t in self.tickers
        })

        self.results = self.backtest_engine.run_backtest(
            self.positions,
            prices
        )

    # ------------------------------------------------
    # PIPELINE
    # ------------------------------------------------

    def run_complete_strategy(self):

        self.step1_collect_data()
        self.step2_features()
        self.step3_ml_training()
        self.step4_regimes()
        self.step5_portfolio()
        self.step6_backtest()

        return self.results