from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.preprocessing import TargetEncoder

from src.features.engineering import (
    feature_engineering,
    RANDOM_STATE,
    NUM_MEDIAN, NUM_ZERO,
    CAT_OHE_MODE, CAT_OHE_MISSING,
    CAT_ORDINAL_MODE, CAT_ORDINAL_MISSING,
    CAT_TARGET,
    ORDINAL_MODE_ORDER, ORDINAL_MISSING_ORDER
)

def get_preprocessing_pipeline():
    """
    Returns the complete preprocessing pipeline including feature engineering
    and column-specific transformations.
    """
    
    # 1. Feature Engineering Step (Custom Function)
    feature_transformer = FunctionTransformer(feature_engineering, validate=False)

    # 2. Individual Transformation Pipes
    
    num_median_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    num_zero_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    cat_ohe_mode_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist', 
                              min_frequency=30, 
                              sparse_output=False))
    ])

    cat_ohe_missing_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist', 
                              min_frequency=30, 
                              sparse_output=False))
    ])

    cat_ord_mode_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('oe', OrdinalEncoder(
            categories=ORDINAL_MODE_ORDER,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    cat_ord_missing_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
        ('oe', OrdinalEncoder(
            categories=ORDINAL_MISSING_ORDER,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    target_encode_pipe = TargetEncoder(
        target_type='continuous',
        cv=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # 3. Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_median', num_median_pipe, NUM_MEDIAN),
            ('num_zero', num_zero_pipe, NUM_ZERO),
            ('cat_ohe_mode', cat_ohe_mode_pipe, CAT_OHE_MODE),
            ('cat_ohe_miss', cat_ohe_missing_pipe, CAT_OHE_MISSING),
            ('cat_ord_mode', cat_ord_mode_pipe, CAT_ORDINAL_MODE),
            ('cat_ord_miss', cat_ord_missing_pipe, CAT_ORDINAL_MISSING),
            ('cat_target', target_encode_pipe, CAT_TARGET)
        ],
        remainder='passthrough'
    )

    # 4. Final Combined Pipeline
    full_pipeline = Pipeline(steps=[
        ('feature_engineering', feature_transformer),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline
