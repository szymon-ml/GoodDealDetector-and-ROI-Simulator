import numpy as np
import pandas as pd

def calculate_flipping_potential(df, y_pred_intrinsic, high_demand_neighborhoods):
    """
    Derives the Flipping_Potential binary target based on:
    1. Underpriced: Predicted intrinsic value > Actual SalePrice
    2. Good Base: Overall Quality >= 6
    3. Market Liquidity: Neighborhood in high demand list
    4. Renovation Potential: At least 2 fixable flaws
    """
    df_ = df.copy()
    
    # 1. Underpriced Condition
    # Note: This requires the actual SalePrice to be present in the df
    if 'SalePrice' in df_.columns:
        # Assuming y_pred_intrinsic is in the same scale as SalePrice (or log)
        # The notebook uses residuals or direct comparison
        df_['Underpriced'] = (y_pred_intrinsic > df_['SalePrice']).astype(int)
    else:
        # If SalePrice is missing, we can't calculate this part
        df_['Underpriced'] = 0

    # 2. Good Base
    c2_good_base = (df_['Overall Qual'] >= 6)

    # 3. Market Liquidity
    c3_liquid_market = df_['Neighborhood'].isin(high_demand_neighborhoods)

    # 4. Renovation Potential (Fixable flaws)
    # Re-calculating TotalBathrooms if not present
    if 'TotalBathrooms' not in df_.columns:
        df_['TotalBathrooms'] = (df_['Full Bath'] + 0.5 * df_['Half Bath'] + 
                                  df_['Bsmt Full Bath'] + 0.5 * df_['Bsmt Half Bath'])

    # Fill NaNs for FireplaceQu
    fireplace_qu = df_['Fireplace Qu'].fillna('No Fireplace')

    sub_c1_kitchen = df_['Kitchen Qual'].isin(['TA', 'Gd'])
    sub_c2_exter = df_['Exter Qual'].isin(['Fa', 'TA', 'Gd'])
    sub_c3_bsmt = df_['Bsmt Qual'].isin(['TA', 'Gd'])
    sub_c4_bath = df_['TotalBathrooms'].isin([1, 1.5, 2.0, 2.5])
    sub_c5_fireplace = fireplace_qu.isin(['TA', 'Gd'])

    renovation_score = (
        sub_c1_kitchen.astype(int) + 
        sub_c2_exter.astype(int) + 
        sub_c3_bsmt.astype(int) + 
        sub_c4_bath.astype(int) +
        sub_c5_fireplace.astype(int)
    )

    c4_renovation_potential = (renovation_score >= 2)

    # Combine All
    flipping_potential = (
        (df_['Underpriced'] == 1) & 
        c2_good_base & 
        c3_liquid_market & 
        c4_renovation_potential
    ).astype(int)

    return flipping_potential
