"""
Utility functions for the crypto trading system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_equity_curve(equity: List[float], title: str = "Equity Curve") -> None:
    """Plot equity curve"""
    plt.figure(figsize=(12, 6))
    plt.plot(equity, linewidth=2, color='blue')
    plt.title(title, fontsize=16)
    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Capital ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    """Plot feature importance"""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_price_and_signals(df: pd.DataFrame, signals_col: str = 'Final_Signal') -> None:
    """Plot price chart with buy signals"""
    plt.figure(figsize=(15, 8))
    
    # Plot price
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'], linewidth=1, color='black', label='Price')
    
    # Mark buy signals
    buy_signals = df[df[signals_col] == 1]
    plt.scatter(buy_signals.index, buy_signals['Close'], 
               color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    
    plt.title('Price and Trading Signals', fontsize=16)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot RSI
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['RSI'], linewidth=1, color='purple', label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
    plt.ylabel('RSI', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def calculate_performance_metrics(equity: List[float], 
                                trades: List[Dict[str, Any]],
                                initial_capital: float) -> Dict[str, float]:
    """Calculate comprehensive performance metrics"""
    if not trades:
        return {}
    
    df_trades = pd.DataFrame(trades)
    
    # Basic metrics
    total_trades = len(df_trades)
    winning_trades = len(df_trades[df_trades['result'] == 'profit'])
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    # Return metrics
    total_return = (equity[-1] / initial_capital - 1) * 100
    
    # Risk metrics
    returns = df_trades['net_pnl'] / initial_capital
    avg_return = returns.mean()
    return_std = returns.std()
    sharpe_ratio = avg_return / return_std if return_std > 0 else 0
    
    # Drawdown
    peaks = np.maximum.accumulate(equity)
    drawdowns = [(equity[i] - peaks[i]) / peaks[i] for i in range(len(equity))]
    max_drawdown = min(drawdowns) if drawdowns else 0
    
    # Average trade metrics
    avg_win = df_trades[df_trades['result'] == 'profit']['net_pnl'].mean() if winning_trades > 0 else 0
    avg_loss = df_trades[df_trades['result'] == 'loss']['net_pnl'].mean() if (total_trades - winning_trades) > 0 else 0
    
    # Profit factor
    gross_profit = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].sum()
    gross_loss = abs(df_trades[df_trades['net_pnl'] < 0]['net_pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'final_capital': equity[-1]
    }


def print_performance_report(metrics: Dict[str, float]) -> None:
    """Print a formatted performance report"""
    print("\n" + "="*50)
    print("           PERFORMANCE REPORT")
    print("="*50)
    
    print(f"Total Trades:        {metrics.get('total_trades', 0)}")
    print(f"Winning Trades:      {metrics.get('winning_trades', 0)}")
    print(f"Win Rate:           {metrics.get('win_rate', 0):.2f}%")
    print(f"Total Return:       {metrics.get('total_return', 0):.2f}%")
    print(f"Final Capital:      ${metrics.get('final_capital', 0):.2f}")
    print(f"Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown:       {metrics.get('max_drawdown', 0):.2f}%")
    print(f"Average Win:        ${metrics.get('avg_win', 0):.2f}")
    print(f"Average Loss:       ${metrics.get('avg_loss', 0):.2f}")
    print(f"Profit Factor:      {metrics.get('profit_factor', 0):.2f}")
    print("="*50)


def save_results_to_csv(trades: List[Dict[str, Any]], 
                       metrics: Dict[str, float],
                       filename: str = 'trading_results.csv') -> None:
    """Save trading results to CSV file"""
    try:
        # Save trades
        if trades:
            df_trades = pd.DataFrame(trades)
            df_trades.to_csv(f'trades_{filename}', index=False)
            logger.info(f"Trades saved to trades_{filename}")
        
        # Save metrics
        df_metrics = pd.DataFrame([metrics])
        df_metrics.to_csv(f'metrics_{filename}', index=False)
        logger.info(f"Metrics saved to metrics_{filename}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")


def validate_config(config_dict: Dict[str, Any]) -> bool:
    """Validate configuration parameters"""
    required_params = [
        'SYMBOLS', 'PROFIT_TARGET', 'STOP_LOSS', 'INITIAL_CAPITAL',
        'RISK_PCT', 'CONFIDENCE_THRESHOLD'
    ]
    
    for param in required_params:
        if param not in config_dict:
            logger.error(f"Missing required parameter: {param}")
            return False
    
    # Validate ranges
    if config_dict['PROFIT_TARGET'] <= 0 or config_dict['PROFIT_TARGET'] > 1:
        logger.error("PROFIT_TARGET must be between 0 and 1")
        return False
    
    if config_dict['STOP_LOSS'] <= 0 or config_dict['STOP_LOSS'] > 1:
        logger.error("STOP_LOSS must be between 0 and 1")
        return False
    
    if config_dict['RISK_PCT'] <= 0 or config_dict['RISK_PCT'] > 0.1:
        logger.error("RISK_PCT must be between 0 and 0.1 (10%)")
        return False
    
    if config_dict['CONFIDENCE_THRESHOLD'] < 50 or config_dict['CONFIDENCE_THRESHOLD'] > 100:
        logger.error("CONFIDENCE_THRESHOLD must be between 50 and 100")
        return False
    
    logger.info("Configuration validation passed")
    return True


def create_summary_dashboard(results: pd.DataFrame) -> None:
    """Create a summary dashboard of results"""
    if results.empty:
        logger.warning("No results to display")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Returns by symbol
    axes[0, 0].bar(results['symbol'], results['total_return_pct'])
    axes[0, 0].set_title('Total Return by Symbol')
    axes[0, 0].set_ylabel('Return (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Win rate by symbol
    axes[0, 1].bar(results['symbol'], results['win_rate_pct'])
    axes[0, 1].set_title('Win Rate by Symbol')
    axes[0, 1].set_ylabel('Win Rate (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Number of trades
    axes[1, 0].bar(results['symbol'], results['total_trades'])
    axes[1, 0].set_title('Number of Trades by Symbol')
    axes[1, 0].set_ylabel('Number of Trades')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Sharpe ratio
    axes[1, 1].bar(results['symbol'], results['sharpe'])
    axes[1, 1].set_title('Sharpe Ratio by Symbol')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()