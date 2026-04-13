import pandas as pd
import joblib
import os
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

# We import our own modules using the new package structure
from src.features.pipeline import get_preprocessing_pipeline
from src.features.engineering import RANDOM_STATE

def train_intrinsic_value_model(X, y, params=None):
    """
    Trains the XGBRegressor for Intrinsic Value estimation.
    Wraps the preprocessing and model into a single Pipeline object.
    Uses TransformedTargetRegressor to handle log-transform of SalePrice internally.
    """
    # Use best params found in research, or defaults
    if params is None:
        params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 3,
            'subsample': 0.7,
            'random_state': RANDOM_STATE
        }

    # Initialize the base preprocessing pipeline
    preprocessor = get_preprocessing_pipeline()
    
    # Initialize the regressor
    regressor = XGBRegressor(**params)
    
    # Wrap regressor to handle log transform of target (SalePrice)
    # This ensures model is trained on logs (best for house prices)
    # but the pipeline.predict() returns actual dollars.
    model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    # Create final model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    print("Training Intrinsic Value Regressor (Log-Transformed Target)...")
    model_pipeline.fit(X, y)
    return model_pipeline

def train_flipping_potential_model(X, y, imbalanced_ratio=1.0, params=None):
    """
    Trains the XGBClassifier for Flipping Potential.
    Handles class imbalance using scale_pos_weight.
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': RANDOM_STATE
        }

    preprocessor = get_preprocessing_pipeline()
    
    # Classifier with imbalance handling
    classifier = XGBClassifier(
        scale_pos_weight=imbalanced_ratio,
        **params
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    print(f"Training Flipping Potential Classifier (Ratio: {imbalanced_ratio:.2f})...")
    model_pipeline.fit(X, y)
    return model_pipeline

def save_model(model, filepath):
    """Saves the fitted model pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Loads a model pipeline from disk."""
    return joblib.load(filepath)
