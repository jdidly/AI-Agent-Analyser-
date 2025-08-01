"""
Machine learning models for crypto trading predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import logging
import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class MLModels:
    """Machine learning models for trading signal prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None
    
    def prepare_data(self, df: pd.DataFrame, features: list) -> tuple:
        """
        Prepare data for machine learning
        
        Args:
            df: DataFrame with features and target
            features: List of feature column names
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        if df is None or df.empty:
            logger.error("DataFrame is None or empty")
            return None, None
        
        # Check if all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features in DataFrame: {missing_features}")
            return None, None
        
        if 'Target' not in df.columns:
            logger.error("Target column not found in DataFrame")
            return None, None
        
        X = df[features].copy()
        y = df['Target'].copy()
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Prepared data: {len(X)} samples, {len(features)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest', tune_hyperparams: bool = True, n_iter: int = 20) -> bool:
        """
        Train the machine learning model
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to train ('random_forest', 'gradient_boosting', 'xgboost', or 'stacking')
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            if len(X) == 0 or len(y) == 0:
                logger.error("Empty dataset provided for training")
                return False
            
            # Check if we have at least 2 classes
            if y.nunique() < 2:
                logger.warning("Only one class in target variable. Training anyway...")
                X_scaled = self.scaler.fit_transform(X)
                self.model = RandomForestClassifier(
                    n_estimators=config.N_ESTIMATORS,
                    max_depth=config.MAX_DEPTH,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                self.model.fit(X_scaled, y)
                self.is_trained = True
                return True
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

            # Stacking ensemble with hyperparameter tuning for base models
            if model_type == 'stacking':
                from sklearn.ensemble import StackingClassifier
                from sklearn.model_selection import RandomizedSearchCV
                # Define base models and their grids
                rf = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1, class_weight='balanced')
                gb = GradientBoostingClassifier(random_state=config.RANDOM_STATE)
                estimators = [('rf', rf), ('gb', gb)]
                param_grids = {
                    'rf': {
                        'n_estimators': [50, 100, 200, 300],
                        'max_depth': [3, 5, 10, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'gb': {
                        'n_estimators': [50, 100, 200, 300],
                        'max_depth': [3, 5, 10],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'subsample': [0.8, 1.0]
                    }
                }
                if xgboost_available:
                    scale_pos_weight = 1.0
                    if y.nunique() == 2:
                        n_pos = (y == 1).sum()
                        n_neg = (y == 0).sum()
                        if n_pos > 0:
                            scale_pos_weight = n_neg / n_pos
                    xgb = XGBClassifier(random_state=config.RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, scale_pos_weight=scale_pos_weight)
                    estimators.append(('xgb', xgb))
                    param_grids['xgb'] = {
                        'n_estimators': [50, 100, 200, 300],
                        'max_depth': [3, 5, 10],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'subsample': [0.8, 1.0]
                    }
                # Tune each base model
                tuned_estimators = []
                for name, est in estimators:
                    grid = param_grids[name]
                    search = RandomizedSearchCV(est, grid, n_iter=30, cv=3, scoring='f1', n_jobs=-1, random_state=config.RANDOM_STATE)
                    search.fit(X_scaled, y)
                    tuned_estimators.append((name, search.best_estimator_))
                    logger.info(f"Tuned {name}: {search.best_params_}")
                logger.info("All base model hyperparameter tuning complete.")
                final_estimator = GradientBoostingClassifier(random_state=config.RANDOM_STATE)
                self.model = StackingClassifier(estimators=tuned_estimators, final_estimator=final_estimator, n_jobs=-1, passthrough=True)
                self.model.fit(X_scaled, y)
                self.is_trained = True
                logger.info("Stacking ensemble (with tuned base models) trained successfully.")
                return True

            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            val_scores = []
            train_scores = []
            best_val_score = -np.inf
            best_model = None
            best_split = None
            logger.info(f"Using TimeSeriesSplit with {tscv.get_n_splits()} splits for validation and cross-validation.")

            # Hyperparameter grids
            rf_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced']
            }
            gb_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0]
            }
            xgb_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0]
            }


            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Initialize model and param grid
                if model_type == 'random_forest':
                    base_model = RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1, class_weight='balanced')
                    param_grid = rf_grid
                elif model_type == 'gradient_boosting':
                    base_model = GradientBoostingClassifier(random_state=config.RANDOM_STATE)
                    param_grid = gb_grid
                elif model_type == 'xgboost':
                    if not xgboost_available:
                        logger.error("XGBoost is not installed. Please install xgboost to use this model.")
                        return False
                    # Compute scale_pos_weight for XGBoost
                    scale_pos_weight = 1.0
                    if y.nunique() == 2:
                        n_pos = (y == 1).sum()
                        n_neg = (y == 0).sum()
                        if n_pos > 0:
                            scale_pos_weight = n_neg / n_pos
                    base_model = XGBClassifier(random_state=config.RANDOM_STATE, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, scale_pos_weight=scale_pos_weight)
                    param_grid = xgb_grid
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                # Hyperparameter tuning
                if tune_hyperparams:
                    from sklearn.model_selection import RandomizedSearchCV
                    search = RandomizedSearchCV(base_model, param_grid, n_iter=30, cv=TimeSeriesSplit(n_splits=3), scoring='f1', n_jobs=-1, random_state=config.RANDOM_STATE)
                    search.fit(X_train, y_train)
                    model = search.best_estimator_
                    logger.info(f"Fold {fold+1}: Best params: {search.best_params_}")
                else:
                    model = base_model
                    model.fit(X_train, y_train)

                train_score = model.score(X_train, y_train)
                val_score = model.score(X_val, y_val)
                train_scores.append(train_score)
                val_scores.append(val_score)
                logger.info(f"Fold {fold+1}: Train acc={train_score:.3f}, Val acc={val_score:.3f}")
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model = model
                    best_split = (train_idx, val_idx)

            if tune_hyperparams:
                logger.info("Hyperparameter tuning for all folds complete.")

            self.model = best_model
            self.is_trained = True

            logger.info(f"Best validation accuracy: {best_val_score:.3f}")
            logger.info(f"Average train accuracy: {np.mean(train_scores):.3f}")
            logger.info(f"Average validation accuracy: {np.mean(val_scores):.3f}")

            # Cross-validation using TimeSeriesSplit
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=tscv, scoring='accuracy')
            logger.info(f"TimeSeriesSplit cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                logger.info("Feature importance:")
                for _, row in self.feature_importance.head().iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.3f}")

            # Classification report (on best validation split)
            if best_split is not None:
                _, val_idx = best_split
                X_val = X_scaled.iloc[val_idx]
                y_val = y.iloc[val_idx]
                y_pred = self.model.predict(X_val)
                logger.info("Classification report:")
                logger.info(f"\n{classification_report(y_val, y_pred)}")

            logger.info("Model training completed successfully")
            return True
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                logger.info("Feature importance:")
                for _, row in self.feature_importance.head().iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.3f}")
            
            # Classification report
            y_pred = self.model.predict(X_val)
            logger.info("Classification report:")
            logger.info(f"\n{classification_report(y_val, y_pred)}")
            
            self.is_trained = True
            logger.info("Model training completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            logger.error("Model is not trained. Call train_model() first.")
            return np.array([])
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return np.array([])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            logger.error("Model is not trained. Call train_model() first.")
            return np.array([])
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)
            return probabilities
        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {str(e)}")
            return np.array([])
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model"""
        if self.feature_importance is None:
            logger.warning("Feature importance not available")
            return pd.DataFrame()
        return self.feature_importance
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model to disk"""
        try:
            import joblib
            if not self.is_trained:
                logger.error("Cannot save untrained model")
                return False
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model from disk"""
        try:
            import joblib
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data.get('feature_importance')
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False