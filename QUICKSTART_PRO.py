"""
CLIMATE-ALPHA v4 QUICKSTART
Run this file to execute the full pipeline
"""

from PRO_MASTER_STRATEGY import ClimateAlphaMasterStrategy


def main():

    TICKERS = ["ICLN", "TAN", "ENPH", "FSLR"]

    strategy = ClimateAlphaMasterStrategy(
        tickers=TICKERS,
        start_date="2019-01-01",
        end_date="2024-12-31"
    )

    results = strategy.run_complete_strategy()

    metrics = results["metrics"]

    print("\nFINAL RESULTS")
    print("-"*40)

    print(f"Annual Return : {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio  : {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown  : {metrics['max_drawdown']:.2%}")
    print(f"Win Rate      : {metrics['win_rate']:.2%}")


if __name__ == "__main__":
    main()