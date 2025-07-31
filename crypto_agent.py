"""
Main Crypto AI Trading Agent
Fixed and improved version of the original notebook
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import logging

# Import our modules
from data_handler import DataHandler
from ml_models import MLModels
from trading_strategy import TradingStrategy
import config

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def run_simulation(
    symbols=None,
    profit_target=None,
    stop_loss=None,
    confidence_threshold=None,
    period=None,
    initial_capital=None,
    risk_pct=None,
    spread=None,
    commission=None
):
    """
    Run the improved crypto trading simulation
    Fixed version of the original function with proper pandas handling
    """
    # Use config defaults if not provided
    symbols = symbols or config.SYMBOLS
    profit_target = profit_target or config.PROFIT_TARGET
    stop_loss = stop_loss or config.STOP_LOSS
    confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
    period = period or config.PERIOD
    initial_capital = initial_capital or config.INITIAL_CAPITAL
    risk_pct = risk_pct or config.RISK_PCT
    spread = spread or config.SPREAD
    commission = commission or config.COMMISSION
    
    results = []
    
    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        
        # Initialize data handler
        data_handler = DataHandler()
        df = data_handler.fetch_data(symbol, period=period, profit_target=profit_target * 100)
        
        if df is None or df.empty:
            logger.error(f"No data for {symbol}")
            continue
        
        # Prepare features and target
        features = data_handler.get_features()
        ml_model = MLModels()
        X, y = ml_model.prepare_data(df, features)
        
        if X is None or y is None:
            logger.error(f"Failed to prepare data for {symbol}")
            continue
        
        # Train model
        if not ml_model.train_model(X, y):
            logger.error(f"Failed to train model for {symbol}")
            continue
        
        # Generate predictions
        predictions = ml_model.predict(X)
        probabilities = ml_model.predict_proba(X)
        
        # Generate trading signals
        strategy = TradingStrategy()
        df_with_signals = strategy.generate_signals(df.loc[X.index], predictions, probabilities)
        
        # Run backtest simulation
        capital = float(initial_capital)
        equity = [capital]
        trades = []
        position = None
        
        for idx in range(len(df_with_signals) - 1):
            row = df_with_signals.iloc[idx]
            nxt = df_with_signals.iloc[idx + 1]
            
            # Check for buy signal
            buy_signal = row['Final_Signal']
            
            if buy_signal == 1 and position is None:
                # Calculate position size based on risk
                risk_amount = capital * risk_pct
                entry_price = float(row['Close']) * (1 + spread)
                stop_price = entry_price * (1 - stop_loss)
                risk_per_unit = entry_price - stop_price
                qty = risk_amount / (risk_per_unit + 1e-9)
                
                # Fixed: Proper handling of pandas Series values
                nxt_high = float(nxt['High'])
                nxt_low = float(nxt['Low'])
                
                # Check exit conditions
                if nxt_high >= entry_price * (1 + profit_target):
                    # Hit profit target
                    exit_price = entry_price * (1 + profit_target) * (1 - spread)
                    result = 'profit'
                elif nxt_low <= stop_price:
                    # Hit stop loss
                    exit_price = stop_price * (1 - spread)
                    result = 'loss'
                else:
                    # No exit condition met
                    continue
                
                # Calculate P&L
                pnl = (exit_price - entry_price) * qty
                fee_cost = commission * (entry_price + exit_price) * qty
                net_pnl = pnl - fee_cost
                
                # Update capital
                capital += net_pnl
                equity.append(capital)
                
                # Record trade
                trades.append({
                    'entry_time': row.name,
                    'exit_time': nxt.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'qty': qty,
                    'net_pnl': net_pnl,
                    'result': result,
                    'confidence': float(row['ML_Confidence']),
                    'signal_strength': float(row['Signal_Strength'])
                })
                
                position = None
        
        # Calculate performance metrics
        df_trades = pd.DataFrame(trades)
        
        if len(df_trades) > 0:
            total_trades = len(df_trades)
            win_rate = df_trades['result'].value_counts(normalize=True).get('profit', 0) * 100
            total_return = (capital / initial_capital - 1) * 100
            
            returns = df_trades['net_pnl'] / initial_capital
            sharpe = returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0
            
            # Calculate drawdown
            peaks = np.maximum.accumulate(equity)
            drawdowns = [(equity[i] - peaks[i]) / peaks[i] for i in range(len(equity))]
            max_dd = min(drawdowns) if drawdowns else 0
            
            # Display results
            print(f"\n=== {symbol} Results ===")
            print(f"Final Capital: ${capital:.2f}")
            print(f"Total Return: {total_return:.2f}%")
            print(f"Total Trades: {total_trades}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            print(f"Max Drawdown: {max_dd:.2%}")
            
            # Plot equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(equity, label='Equity Curve', linewidth=2)
            plt.title(f'{symbol} - Equity Curve')
            plt.xlabel('Trade Number')
            plt.ylabel('Capital ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
            # Store results
            results.append({
                'symbol': symbol,
                'final_capital': capital,
                'total_return_pct': total_return,
                'total_trades': total_trades,
                'win_rate_pct': win_rate,
                'sharpe': sharpe,
                'max_drawdown_pct': max_dd * 100
            })
        else:
            logger.warning(f"No trades generated for {symbol}")
            results.append({
                'symbol': symbol,
                'final_capital': initial_capital,
                'total_return_pct': 0,
                'total_trades': 0,
                'win_rate_pct': 0,
                'sharpe': 0,
                'max_drawdown_pct': 0
            })
    
    return pd.DataFrame(results)


def main():
    """Main function to run the crypto trading agent"""
    logger.info("Starting Crypto AI Trading Agent...")
    
    # Run simulation with improved parameters
    results = run_simulation()
    
    if not results.empty:
        print("\n=== SUMMARY RESULTS ===")
        print(results.to_string(index=False))
        
        # Save results
        results.to_csv('trading_results.csv', index=False)
        logger.info("Results saved to trading_results.csv")
    else:
        logger.error("No results generated")


if __name__ == "__main__":
    main()