"""
Main Crypto AI Trading Agent
Fixed and improved version of the original notebook
"""

import requests
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
    # Lower profit target to increase positive samples
    profit_target = 0.005  # 0.5% profit target
    stop_loss = stop_loss or config.STOP_LOSS
    confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
    period = period or config.PERIOD
    initial_capital = initial_capital or config.INITIAL_CAPITAL
    risk_pct = risk_pct or config.RISK_PCT
    spread = spread or config.SPREAD
    commission = commission or config.COMMISSION
    
    results = []
    


    from sklearn.metrics import f1_score
    window_size = 200  # Number of samples for initial training window
    step_size = 20     # Step size for walk-forward


    for symbol in symbols:
        logger.info(f"Processing {symbol} with walk-forward validation...")

        # Initialize data handler
        data_handler = DataHandler()
        df = data_handler.fetch_data(symbol, period=period, profit_target=profit_target * 100)

        if df is None or df.empty:
            logger.error(f"No data for {symbol}")
            continue

        features = data_handler.get_features()
        X_full = df[features]
        y = df['Target']

        # --- OPTIMIZED: Tune and select features ONCE per symbol ---
        N_FEATURES = 8
        fs_model = MLModels()
        logger.info("Tuning hyperparameters and selecting features on initial window...")
        # Use only the first window for tuning
        initial_X = X_full.iloc[:window_size]
        initial_y = y.iloc[:window_size]
        tuned = fs_model.train_model(initial_X, initial_y, model_type='stacking', tune_hyperparams=True)
        importances = None
        if tuned and hasattr(fs_model.model, 'estimators_') and hasattr(fs_model.model, 'final_estimator_') and hasattr(fs_model.model.final_estimator_, 'feature_importances_'):
            importances = fs_model.model.final_estimator_.feature_importances_
        elif tuned and hasattr(fs_model.model, 'feature_importances_'):
            importances = fs_model.model.feature_importances_
        if importances is not None:
            if hasattr(fs_model.model, 'feature_names_in_'):
                model_features = list(fs_model.model.feature_names_in_)
            else:
                model_features = features
            feat_imp = list(zip(model_features, importances))
            feat_imp = sorted(feat_imp, key=lambda x: x[1], reverse=True)
            selected_features = [f for f, _ in feat_imp[:min(N_FEATURES, len(feat_imp))] if f in X_full.columns]
            logger.info(f"Selected features: {selected_features}")
        else:
            logger.warning("Feature selection skipped: could not determine importances. Using all features.")
            selected_features = features
        X = X_full[selected_features]

        n_samples = len(df)
        equity = []
        capital = float(initial_capital)
        equity.append(capital)
        trades = []
        position = None

        # --- Use best params/features for all walk-forward windows ---
        for start in range(0, n_samples - window_size - 1, step_size):
            end = start + window_size
            X_train, y_train = X.iloc[start:end], y.iloc[start:end]
            X_test, y_test = X.iloc[end:end+step_size], y.iloc[end:end+step_size]
            if len(X_test) == 0:
                break

            # Use the tuned model's best params for all windows
            ml_model = MLModels()
            # Use best params from fs_model if available
            if hasattr(fs_model.model, 'get_params'):
                best_params = fs_model.model.get_params()
            else:
                best_params = None
            # Train with best params, but no further tuning
            ml_model.train_model(X_train, y_train, model_type='stacking', tune_hyperparams=False)

            proba = ml_model.predict_proba(X_train)
            if proba is None or len(proba) == 0:
                continue
            best_thr = confidence_threshold
            best_f1 = -np.inf
            for thr in np.arange(50, 95, 2.5):
                if len(proba.shape) == 1 or (len(proba.shape) == 2 and proba.shape[1] == 1):
                    conf = proba[:, 0] * 100
                else:
                    conf = proba[:, 1] * 100
                preds = (conf >= thr).astype(int)
                f1 = f1_score(y_train, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
            proba_test = ml_model.predict_proba(X_test)
            if proba_test is None or len(proba_test) == 0:
                continue
            if len(proba_test.shape) == 1 or (len(proba_test.shape) == 2 and proba_test.shape[1] == 1):
                conf_test = proba_test[:, 0] * 100
            else:
                conf_test = proba_test[:, 1] * 100
            buy_signals = (conf_test >= best_thr).astype(int)
            for i, idx in enumerate(range(end, min(end+step_size, n_samples-1))):
                buy_signal = buy_signals[i]
                row = df.iloc[idx]
                nxt = df.iloc[idx + 1]
                if buy_signal == 1 and position is None:
                    risk_amount = capital * risk_pct
                    entry_price = row['Close'] * (1 + spread)
                    stop_price = entry_price * (1 - stop_loss)
                    risk_per_unit = entry_price - stop_price
                    qty = risk_amount / (risk_per_unit + 1e-9)
                    nxt_high = float(nxt['High'])
                    nxt_low = float(nxt['Low'])
                    if nxt_high >= entry_price * (1 + profit_target):
                        exit_price = entry_price * (1 + profit_target) * (1 - spread)
                        result = 'profit'
                    elif nxt_low <= stop_price:
                        exit_price = stop_price * (1 - spread)
                        result = 'loss'
                    else:
                        continue
                    pnl = (exit_price - entry_price) * qty
                    fee_cost = commission * (entry_price + exit_price) * qty
                    net_pnl = pnl - fee_cost
                    capital += net_pnl
                    equity.append(capital)
                    trades.append({
                        'entry_time': row.name,
                        'exit_time': nxt.name,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'qty': qty,
                        'net_pnl': net_pnl,
                        'result': result,
                        'confidence': conf_test[i]
                    })
                    position = None

        df_trades = pd.DataFrame(trades)
        total_trades = len(df_trades)
        win_rate = df_trades['result'].value_counts(normalize=True).get('profit', 0) * 100 if total_trades > 0 else 0
        total_return = (capital / initial_capital - 1) * 100
        returns = df_trades['net_pnl'] / initial_capital if total_trades > 0 else pd.Series([0])
        sharpe = returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0
        peaks = np.maximum.accumulate(equity)
        drawdowns = [(equity[i] - peaks[i]) / peaks[i] for i in range(len(equity))]
        max_dd = min(drawdowns) if drawdowns else 0

        # Custom metrics: F1 and precision for buy signals in test windows
        all_true = []
        all_pred = []
        for start in range(0, n_samples - window_size - 1, step_size):
            end = start + window_size
            X_train, y_train = X.iloc[start:end], y.iloc[start:end]
            X_test, y_test = X.iloc[end:end+step_size], y.iloc[end:end+step_size]
            if len(X_test) == 0:
                break
            ml_model = MLModels()
            trained = ml_model.train_model(X_train, y_train, model_type='stacking', tune_hyperparams=False)
            if not trained:
                continue
            proba_test = ml_model.predict_proba(X_test)
            if proba_test is None or len(proba_test) == 0:
                continue
            if len(proba_test.shape) == 1 or (len(proba_test.shape) == 2 and proba_test.shape[1] == 1):
                conf_test = proba_test[:, 0] * 100
            else:
                conf_test = proba_test[:, 1] * 100
            preds = (conf_test >= confidence_threshold).astype(int)
            all_true.extend(list(y_test))
            all_pred.extend(list(preds))
        from sklearn.metrics import f1_score, precision_score
        f1 = f1_score(all_true, all_pred) if all_true else 0
        precision = precision_score(all_true, all_pred) if all_true else 0

        print(f"\n=== {symbol} Walk-Forward Results ===")
        print(f"Final Capital: ${capital:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"F1 Score: {f1:.3f}")
        print(f"Precision: {precision:.3f}")

        plt.figure(figsize=(12, 6))
        plt.plot(equity, label='Equity Curve', linewidth=2)
        plt.title(f'{symbol} - Walk-Forward Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        results.append({
            'symbol': symbol,
            'final_capital': capital,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'sharpe': sharpe,
            'max_drawdown_pct': max_dd * 100,
            'f1_score': f1,
            'precision': precision
        })

        # (Old post-walk-forward trading loop removed; only walk-forward results are used)
    return pd.DataFrame(results)


def main():
    """Main function to run the crypto trading agent"""
    logger.info("Starting Crypto AI Trading Agent...")
    
    # Run simulation with improved parameters
    results = run_simulation()
    
    if not results.empty:
        print("\n=== SUMMARY RESULTS ===")
        print(results.to_string(index=False))

        # Add timestamp column to results
        import datetime
        results['timestamp'] = datetime.datetime.now().isoformat()

        # Save results (append if file exists, else create with header)
        import os
        file_exists = os.path.isfile('trading_results.csv')
        results.to_csv('trading_results.csv', mode='a', header=not file_exists, index=False)
        logger.info("Results appended to trading_results.csv")
    else:
        logger.error("No results generated")


if __name__ == "__main__":
    main()