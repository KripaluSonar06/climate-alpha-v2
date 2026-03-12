"""
QUICK START - PROFESSIONAL CLIMATE-ALPHA
Just run this file to execute the complete strategy!
"""

from PRO_MASTER_STRATEGY import ClimateAlphaMasterStrategy

def main():
    """
    Run the complete professional Climate-Alpha strategy
    """
    
    print("\n" + "="*80)
    print("CLIMATE-ALPHA PROFESSIONAL QUANTITATIVE SYSTEM")
    print("Hedge-Fund-Grade ESG Trading Platform")
    print("="*80)
    
    # Configuration
    TICKERS = ['ICLN', 'TAN', 'ENPH', 'FSLR']  # Clean energy tickers
    START_DATE = '2019-01-01'
    END_DATE = '2024-12-31'
    PORTFOLIO_METHOD = 'kelly'  # Options: 'kelly', 'risk_parity', 'mean_variance'
    
    print(f"\nConfiguration:")
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Portfolio Method: {PORTFOLIO_METHOD.upper()}")
    
    try:
        # Initialize strategy
        print("\nInitializing strategy...")
        strategy = ClimateAlphaMasterStrategy(
            tickers=TICKERS,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        # Run complete strategy
        print("\nExecuting complete strategy pipeline...")
        results = strategy.run_complete_strategy(portfolio_method=PORTFOLIO_METHOD)
        
        # Compare to benchmark
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON")
        print("="*80)
        strategy.compare_to_benchmark('SPY')
        
        # Final summary
        print("\n" + "="*80)
        print("✓ STRATEGY EXECUTION COMPLETE!")
        print("="*80)
        
        metrics = results['metrics']
        print(f"\nKEY RESULTS:")
        print(f"  Annual Return:    {metrics['annualized_return']:>10.2%}")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Max Drawdown:     {metrics['max_drawdown']:>10.2%}")
        print(f"  Win Rate:         {metrics['win_rate']:>10.2%}")
        print(f"  Profit Factor:    {metrics['profit_factor']:>10.2f}")
        
        # Check if results are good
        is_good = (
            metrics['sharpe_ratio'] > 1.0 and
            metrics['annualized_return'] > 0.12 and
            metrics['max_drawdown'] > -0.30
        )
        
        if is_good:
            print("\n✓ EXCELLENT RESULTS! Ready for IBM submission!")
        else:
            print("\n⚠ Results could be better. Try adjusting parameters.")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Check that all dependencies are installed")
        print("2. Ensure internet connection for data download")
        print("3. Check that all .py files are in the same directory")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    
    if results:
        print("\n" + "="*80)
        print("Next steps:")
        print("1. Save these results to your concept note")
        print("2. Create visualizations for your presentation")
        print("3. Update your resume with these metrics")
        print("4. Submit to IBM for certification!")
        print("="*80)
