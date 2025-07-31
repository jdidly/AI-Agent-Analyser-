"""
Machine learning models for crypto trading predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
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
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest') -> bool:
        """
        Train the machine learning model
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to train ('random_forest' or 'gradient_boosting')
            
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
                self.model = RandomForestClassifier(
                    n_estimators=config.N_ESTIMATORS,
                    max_depth=config.MAX_DEPTH,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1
                )
                self.model.fit(X, y)
                self.is_trained = True
                return True
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, 
                test_size=config.TEST_SIZE, 
                random_state=config.RANDOM_STATE,
                stratify=y
            )
            
            # Initialize model
            if model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=config.N_ESTIMATORS,
                    max_depth=config.MAX_DEPTH,
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1,
                    class_weight='balanced'  # Handle class imbalance
                )
            elif model_type == 'gradient_boosting':
                self.model = GradientBoostingClassifier(
                    n_estimators=config.N_ESTIMATORS,
                    max_depth=config.MAX_DEPTH,
                    random_state=config.RANDOM_STATE
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train model
            logger.info(f"Training {model_type} model...")
            self.model.fit(X_train, y_train)
            
            # Validate model
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)
            
            logger.info(f"Training accuracy: {train_score:.3f}")
            logger.info(f"Validation accuracy: {val_score:.3f}")
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='accuracy')
            logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
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