import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_and_encode_data(
    df: pd.DataFrame, 
    imputation_values: dict[str, float] | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Performs data cleaning, encoding, and imputation.

    Logic implemented:
    - **Ordinal Encoding**: Maps 'grade' (A-G) to integers.
    - **One-Hot Encoding**: Converts 'loan_purpose' to binary vectors.
    - **Imputation**: Uses Median values calculated ONLY from the training set.
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

    # 1. Drop ID column (Irrelevant/Noise)
    if 'id' in df_clean.columns:
        df_clean = df_clean.drop(columns=['id'])

    # 2. Robust Grade Handling
    # If 'grade' is missing but 'grade_subgrade' exists, extract the letter.
    if 'grade' not in df_clean.columns and 'grade_subgrade' in df_clean.columns:
        df_clean['grade'] = df_clean['grade_subgrade'].str[0]
        
    # 3. Ordinal Encoding for 'grade'
    # We map A (best) -> 1 down to G (worst) -> 7.
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    
    if 'grade' in df_clean.columns:
        df_clean['grade_encoded'] = df_clean['grade'].map(grade_map)
        # Fill any grades that didn't match (e.g., NaNs) with a default or mode
        if df_clean['grade_encoded'].isnull().any():
             df_clean['grade_encoded'] = df_clean['grade_encoded'].fillna(4) # Middle grade D
        df_clean = df_clean.drop(columns=['grade'])
    
    # Drop subgrade to avoid multicollinearity (since we captured the info in grade)
    if 'grade_subgrade' in df_clean.columns:
        df_clean = df_clean.drop(columns=['grade_subgrade'])

    # 4. One-Hot Encoding for 'loan_purpose'
    # drop_first=True prevents the "Dummy Variable Trap" (Multicollinearity).
    if 'loan_purpose' in df_clean.columns:
        df_clean = pd.get_dummies(
            df_clean, 
            columns=['loan_purpose'], 
            drop_first=True, 
            dtype=int
        )

    # 5. Imputation (Anti-Leakage Logic)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # TRAIN MODE: Calculate medians and save them
    if imputation_values is None:
        imputation_values = {}
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                imputation_values[col] = median_val
                df_clean[col] = df_clean[col].fillna(median_val)
    
    # TEST MODE: Use provided medians
    else:
        for col, val in imputation_values.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(val)

    return df_clean, imputation_values

def scale_features(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applies Standard Scaling (Z-score) to numeric features.
    
    Strictly follows the "Fit on Train, Transform on Test" rule to prevent
    Data Leakage.

    Args:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation/Test features.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Scaled versions of input dataframes.
    """
    scaler = StandardScaler()
    
    # Identify numeric columns (excluding boolean/binary dummies if desired, 
    # but scaling everything is generally safe for these models)
    cols_to_scale = [
        col for col in X_train.columns 
        if X_train[col].nunique() > 2
    ]

    # 1. Fit ONLY on Training data
    scaler.fit(X_train[cols_to_scale])

    # 2. Transform Training data
    X_train_scaled = X_train.copy()
    X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])

    # 3. Transform Validation data using the TRAIN scaler
    X_val_scaled = X_val.copy()
    X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])

    return X_train_scaled, X_val_scaled