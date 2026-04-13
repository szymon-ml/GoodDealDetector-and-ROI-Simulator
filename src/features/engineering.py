import pandas as pd
import numpy as np

RANDOM_STATE = 42

def feature_engineering(df):
    """
    Core feature engineering steps for the Ames Housing dataset.
    Includes log transformations, binary flags for missingness, and interaction terms.
    """
    df_ = df.copy()

    # changing feature types
    df_['MS SubClass'] = df_['MS SubClass'].astype('object')
    df_['Mo Sold'] = df_['Mo Sold'].astype('object')

    # log transforming variables
    high_skew_features = [
        'Lot Frontage', 'Lot Area', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 
        'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 
        'Gr Liv Area', 'Bsmt Half Bath', 'Kitchen AbvGr', 'Wood Deck SF', 
        'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 
        'Pool Area', 'Misc Val'
    ]

    for item in high_skew_features:
        if item in df_.columns:
            df_[item] = np.log1p(df_[item])

    # turning high-missingness features into binaries
    high_missingness_list = ['Pool QC', 'Misc Feature', 'Alley', 'Fence', 'Mas Vnr Type', 'Fireplace Qu']

    for item in high_missingness_list:
        if item in df_.columns:
            df_[item] = df_[item].notna().astype(int) 

    # creating variables through interaction terms
    if all(col in df_.columns for col in ['Total Bsmt SF', '1st Flr SF', '2nd Flr SF']):
        df_['TotalSF'] = df_['Total Bsmt SF'] + df_['1st Flr SF'] + df_['2nd Flr SF']
    
    if all(col in df_.columns for col in ['Yr Sold', 'Year Built']):
        df_['AgeAtSale'] = df_['Yr Sold'] - df_['Year Built']
    
    if all(col in df_.columns for col in ['Yr Sold', 'Year Remod/Add']):
        df_['TimeSinceRemod'] = df_['Yr Sold'] - df_['Year Remod/Add']
    
    if all(col in df_.columns for col in ['Full Bath', 'Half Bath', 'Bsmt Full Bath', 'Bsmt Half Bath']):
        df_['TotalBathrooms'] = (df_['Full Bath'] + 0.5 * df_['Half Bath'] +
                                    df_['Bsmt Full Bath'] + 0.5 * df_['Bsmt Half Bath'])

    # defining new binaries
    if 'Garage Area' in df_.columns:
        df_['HasGarage'] = (df_['Garage Area'] > 0).astype(int)
    if '2nd Flr SF' in df_.columns:
        df_['Has2ndFloor'] = (df_['2nd Flr SF'] > 0).astype(int)
    if all(col in df_.columns for col in ['Year Remod/Add', 'Year Built']):
        df_['WasRemodeled'] = (df_['Year Remod/Add'] != df_['Year Built']).astype(int)

    # dropping unusable features
    if 'Garage Yr Blt' in df_.columns:
        df_ = df_.drop('Garage Yr Blt', axis=1)

    return df_

# --- FEATURE LISTS ---

# 1. Numerical: Impute with median
NUM_MEDIAN = [
    'Year Built', 'Year Remod/Add', 'Overall Qual', 'Overall Cond', 
    '1st Flr SF', '2nd Flr SF', 'Gr Liv Area', 'Full Bath', 'Half Bath', 
    'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd',
    'TotalSF', 'AgeAtSale', 'TimeSinceRemod', 'TotalBathrooms'
]

# 2. Numerical: Impute with 0
NUM_ZERO = [
    'Bsmt Half Bath', 'Garage Area', 'BsmtFin SF 1', 'Bsmt Full Bath', 
    'Bsmt Unf SF', 'Screen Porch', 'Lot Area', 'Garage Cars', 'Fireplaces', 
    'Mas Vnr Area', 'Total Bsmt SF', 'Lot Frontage', 'Wood Deck SF', 
    'BsmtFin SF 2', 'Yr Sold', 'Open Porch SF', '3Ssn Porch', 'Low Qual Fin SF', 
    'Misc Val', 'Enclosed Porch', 'Pool Area'
]

# 3. Categorical One-Hot: Impute with mode
CAT_OHE_MODE = [
    'Electrical', 'Street', 'Land Contour', 'Condition 1', 'Bldg Type', 
    'Central Air', 'Heating', 'Land Slope', 'MS Zoning', 'Roof Matl', 
    'Sale Condition', 'Lot Config', 'Roof Style', 'Lot Shape', 'House Style', 
    'Paved Drive', 'Foundation', 'Condition 2', 'MS SubClass', 'Mo Sold'
]

# 4. Categorical One-Hot: Impute with 'Missing'
CAT_OHE_MISSING = [
    'Pool QC', 'Misc Feature', 'Alley', 'Garage Finish', 'Garage Type', 
    'Fence', 'Mas Vnr Type'
]

# 5. Categorical Ordinal: Impute with mode
CAT_ORDINAL_MODE = [
    'Utilities', 'Heating QC', 'Exter Cond', 'Exter Qual', 'Kitchen Qual', 'Functional'
]

# 6. Categorical Ordinal: Impute with 'None'
CAT_ORDINAL_MISSING = [
    'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 
    'BsmtFin Type 2', 'Fireplace Qu', 'Garage Qual', 'Garage Cond'
]

# 7. Categorical Target:
CAT_TARGET = ['Neighborhood', 'Exterior 1st', 'Exterior 2nd', 'Sale Type']

# --- ORDINAL CATEGORIES ORDER ---

QUAL_FEATURES = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
UTILITIES = ['ELO', 'NoSeWa', 'NoSewr', 'AllPub']
FUNCTIONAL = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']
EXPOSURE = ['None', 'No', 'Mn', 'Av', 'Gd']
BSMTFIN = ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']

ORDINAL_MODE_ORDER = [
    UTILITIES,      # Utilities
    QUAL_FEATURES,  # Heating QC
    QUAL_FEATURES,  # Exter Cond
    QUAL_FEATURES,  # Exter Qual
    QUAL_FEATURES,  # Kitchen Qual
    FUNCTIONAL      # Functional
]

ORDINAL_MISSING_ORDER = [
    QUAL_FEATURES, # Bsmt Qual
    QUAL_FEATURES, # Bsmt Cond
    EXPOSURE,      # Bsmt Exposure
    BSMTFIN,       # BsmtFin Type 1
    BSMTFIN,       # BsmtFin Type 2
    QUAL_FEATURES, # Fireplace Qu
    QUAL_FEATURES, # Garage Qual
    QUAL_FEATURES  # Garage Cond
]
