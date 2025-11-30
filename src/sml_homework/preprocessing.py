import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

def clean_and_encode_data(
    df: pd.DataFrame, 
    imputation_values: dict[str, float] | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Performs data cleaning, encoding, and imputation.

    Logic implemented:
    - **Ordinal Encoding**: Maps 'grade' (A-G) to integers.
    - **Feature Extraction**: Creates 'grade' from 'grade_subgrade'.
    - **One-Hot Encoding**: Converts 'loan_purpose' to binary vectors.
    - **Imputation**: Uses Median values calculated only from the training set.
    - **Leakage Prevention**: If `imputation_values` are provided, it uses them 
      instead of recalculating statistics.

    Args:
        df (pd.DataFrame): The raw dataframe.
        imputation_values (dict[str, float] | None): 
            - If None: Calculates medians from this data (Train mode).
            - If provided: Uses these values to fill NaNs (Test mode).

    Returns:
        tuple[pd.DataFrame, dict[str, float]]: The processed DataFrame and the 
        dictionary of imputation values used.
    """
    df_clean = df.copy()

    # 1. Drop ID column since it is irrelevant
    if 'id' in df_clean.columns:
        df_clean = df_clean.drop(columns=['id'])

    # 2. Reduce data sparsity by extracting 'grade' from 'grade_subgrade'
    # We must extract 'grade' from 'grade_subgrade' before we drop 'grade_subgrade'.
    if 'grade' not in df_clean.columns and 'grade_subgrade' in df_clean.columns:
        df_clean['grade'] = df_clean['grade_subgrade'].str[0]
        
    # 3. Ordinal Encoding for 'grade'
    # We map A (best) -> 1 down to G (worst) -> 7.
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    
    if 'grade' in df_clean.columns:
        df_clean['grade_encoded'] = df_clean['grade'].map(grade_map)
        
        # Handle potential mapping failures (NaNs) by filling with a middle ground (D=4)
        if df_clean['grade_encoded'].isnull().any():
             df_clean['grade_encoded'] = df_clean['grade_encoded'].fillna(4)
             
        # Drop the original 'grade' column now that we have the encoded version
        df_clean = df_clean.drop(columns=['grade'])
    
    # 4. Avoid Multicollinearity
    if 'grade_subgrade' in df_clean.columns and 'grade_encoded' in df_clean.columns:
        df_clean = df_clean.drop(columns=['grade_subgrade'])

    # 5. One-Hot Encoding for 'loan_purpose'
    if 'loan_purpose' in df_clean.columns:
        df_clean = pd.get_dummies(
            df_clean, 
            columns=['loan_purpose'], 
            drop_first=True, 
            dtype=int
        )

    # 6. Imputation to avoid Data Leakage from valid to train
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # Train mode: Calculate medians and save them
    if imputation_values is None:
        imputation_values = {}
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                imputation_values[col] = median_val
                df_clean[col] = df_clean[col].fillna(median_val)
    
    # Test mode: Use provided medians
    else:
        for col, val in imputation_values.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(val)

    # 7. Log Transformation (EDA Finding: Skewness)
    # 'annual_income' showed extreme right-skewness. 
    # We apply log1p (log(x+1)) to normalize the distribution.
    if 'annual_income' in df_clean.columns:
        # Ensure no negative values (though income shouldn't be negative)
        df_clean['annual_income'] = df_clean['annual_income'].clip(lower=0)
        df_clean['annual_income'] = np.log1p(df_clean['annual_income'])

    return df_clean, imputation_values

def scale_features(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies Standard Scaling (Z-score) to numeric features.

    Args:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation/Test features.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Scaled versions of input dataframes.
    """
    scaler = StandardScaler()
    
    # Identify numeric columns
    cols_to_scale = [
        col for col in X_train.columns 
        if X_train[col].nunique() > 2
    ]

    # 1. Fit only on Training data
    scaler.fit(X_train[cols_to_scale])

    # 2. Transform Training data
    X_train_scaled = X_train.copy()
    X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])

    # 3. Transform Validation data using the train scaler
    X_val_scaled = X_val.copy()
    X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])

    return X_train_scaled, X_val_scaled