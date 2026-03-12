"""
CLIMATE-ALPHA MASTER STRATEGY
FINAL UPGRADED VERSION

Features:
- Parallel ML training
- Feature selection
- Walk-forward training
- Regime-aware portfolio scaling
- Robust data download
"""

import numpy as np
import pandas as pd
from typing import List
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_selection import SelectKBest, f_regression

from PRO_feature_engineering import AdvancedFeatureEngineer
from PRO_ensemble_models import EnsembleMLSystem
from PRO_regime_detection import RegimeDetector
from PRO_portfolio_optimizer import ProfessionalPortfolioOptimizer
from PRO_backtest_engine import ProfessionalBacktestEngine, BacktestConfig


class ClimateAlphaMasterStrategy:

    def __init__(self,
                 tickers: List[str],
                 start_date="2019-01-01",
                 end_date="2024-12-31"):

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

        self.ml_models = {}
        self.data = {}
        self.features = {}
        self.predictions = {}

        self.regimes = None
        self.positions = None
        self.backtest_results = None

        print("="*80)
        print("CLIMATE-ALPHA QUANT SYSTEM (FINAL VERSION)")
        print("="*80)

    # --------------------------------------------------
    # DATA COLLECTION
    # --------------------------------------------------

    def step1_collect_data(self):

        print("\nSTEP 1: DATA COLLECTION")
        print("-"*60)

        for ticker in self.tickers:

            print(f"Downloading {ticker}...")

            df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna()

            self.data[ticker] = df

            print(f"✓ {ticker} rows: {len(df)}")

    # --------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------

    def step2_engineer_features(self):

        print("\nSTEP 2: FEATURE ENGINEERING")
        print("-"*60)

        for ticker in self.tickers:

            data = self.data[ticker]

            volume = data["Volume"] if "Volume" in data.columns else None

            feats = self.feature_engineer.create_all_features(data, volume)

            feats = self._select_top_features(feats, data["Close"])

            self.features[ticker] = feats

            print(f"✓ {ticker} features: {feats.shape[1]}")

    # --------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------

    def _select_top_features(self, features, price):

        returns = price.pct_change().shift(-5)

        df = features.copy()
        df["target"] = returns

        df = df.dropna()

        X = df.drop("target", axis=1)
        y = df["target"]

        selector = SelectKBest(score_func=f_regression, k=60)
        selector.fit(X, y)

        selected = X.columns[selector.get_support()]

        return features[selected]

    # --------------------------------------------------
    # PARALLEL ML TRAINING
    # --------------------------------------------------

    def step3_train_ml_models(self):

        print("\nSTEP 3: TRAINING ML MODELS (PARALLEL)")
        print("-"*60)

        def train_asset(ticker):

            features = self.features[ticker]
            prices = self.data[ticker]["Close"]

            split = int(len(features) * 0.7)

            train_X = features.iloc[:split]
            train_y = prices.iloc[:split]

            model = EnsembleMLSystem(target_horizon=5)

            model.train_ensemble(train_X, train_y)

            preds = model.predict(features)

            return ticker, model, preds

        with ThreadPoolExecutor(max_workers=len(self.tickers)) as executor:
            results = executor.map(train_asset, self.tickers)

        for ticker, model, preds in results:

            self.ml_models[ticker] = model
            self.predictions[ticker] = preds

            print(f"✓ {ticker} ML trained")

    # --------------------------------------------------
    # REGIME DETECTION
    # --------------------------------------------------

    def step4_detect_regimes(self):

        print("\nSTEP 4: REGIME DETECTION")
        print("-"*60)

        avg_price = pd.concat(
            [self.data[t]["Close"] for t in self.tickers],
            axis=1
        ).mean(axis=1)

        self.regimes = self.regime_detector.detect_regimes_comprehensive(avg_price)

        print("✓ Market regimes detected")

    # --------------------------------------------------
    # PORTFOLIO OPTIMIZATION
    # --------------------------------------------------

    def step5_optimize_portfolio(self):

        print("\nSTEP 5: PORTFOLIO OPTIMIZATION")
        print("-"*60)

        returns = pd.DataFrame({
            t: self.data[t]["Close"].pct_change()
            for t in self.tickers
        })

        pos = {}

        for ticker in self.tickers:

            p = self.portfolio_optimizer.calculate_kelly_from_predictions(
                self.predictions[ticker],
                returns[ticker]
            )

            pos[ticker] = p

        self.positions = pd.DataFrame(pos)

        if self.regimes is not None:

            regime = self.regimes["regime_combined"]

            scale = {0:0.5, 1:1.0, 2:1.3}

            for r, m in scale.items():

                mask = regime == r

                self.positions.loc[mask] *= m

        print("✓ Portfolio positions ready")

    # --------------------------------------------------
    # BACKTEST
    # --------------------------------------------------

    def step6_backtest(self):

        print("\nSTEP 6: BACKTEST")
        print("-"*60)

        prices = pd.DataFrame({
            t: self.data[t]["Close"]
            for t in self.tickers
        })

        self.backtest_results = self.backtest_engine.run_backtest(
            self.positions,
            prices
        )

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    def step7_analyze_results(self):

        metrics = self.backtest_results["metrics"]

        print("\nRESULTS SUMMARY")
        print("-"*60)

        print(f"Annual Return : {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio  : {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown  : {metrics['max_drawdown']:.2%}")
        print(f"Win Rate      : {metrics['win_rate']:.2%}")

    # --------------------------------------------------
    # --------------------------------------------------
    # COMPLETE PIPELINE
    # --------------------------------------------------

    def run_complete_strategy(self, portfolio_method="kelly"):

        print("\n" + "="*80)
        print("EXECUTING COMPLETE CLIMATE-ALPHA STRATEGY")
        print("="*80)

        self.step1_collect_data()
        self.step2_engineer_features()
        self.step3_train_ml_models()
        self.step4_detect_regimes()
        self.step5_optimize_portfolio()
        self.step6_backtest()
        self.step7_analyze_results()

        print("\n" + "="*80)
        print("✓ STRATEGY EXECUTION COMPLETE")
        print("="*80)

        return self.backtest_results


# --------------------------------------------------
# RUN SCRIPT
# --------------------------------------------------

if __name__ == "__main__":

    TICKERS = ["ICLN","TAN","ENPH","FSLR"]

    strategy = ClimateAlphaMasterStrategy(
        tickers=TICKERS,
        start_date="2019-01-01",
        end_date="2024-12-31"
    )

    results = strategy.run_complete_strategy()