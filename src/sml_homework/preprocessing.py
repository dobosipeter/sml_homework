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
    - **Feature Extraction**: Recovers 'grade' from 'grade_subgrade' if missing.
    - **One-Hot Encoding**: Converts ALL remaining categorical columns (like gender, marital_status) to binary vectors.
    - **Imputation**: Uses Median values calculated ONLY from the training set.
    - **Log Transformation**: Applied to 'annual_income' to handle skewness found in EDA.
    - **Feature Engineering**: Adds 'high_interest' flag based on EDA scatter plot findings.
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

    # 2. Robust Grade Handling
    if 'grade' not in df_clean.columns and 'grade_subgrade' in df_clean.columns:
        df_clean['grade'] = df_clean['grade_subgrade'].str[0]
        
    # 3. Ordinal Encoding for 'grade'
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    
    if 'grade' in df_clean.columns:
        df_clean['grade_encoded'] = df_clean['grade'].map(grade_map)
        if df_clean['grade_encoded'].isnull().any():
             df_clean['grade_encoded'] = df_clean['grade_encoded'].fillna(4)
        df_clean = df_clean.drop(columns=['grade'])
    
    # 4. Remove Redundancy
    if 'grade_encoded' in df_clean.columns and 'grade_subgrade' in df_clean.columns:
        df_clean = df_clean.drop(columns=['grade_subgrade'])

    # 5. One-Hot Encoding for ALL Categorical Columns
    # We identify columns that are still 'object' or 'category' type.
    # This covers: loan_purpose, gender, marital_status, education_level, employment_status
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        df_clean = pd.get_dummies(
            df_clean, 
            columns=categorical_cols, 
            drop_first=True, 
            dtype=int
        )

    # 6. Imputation againsy leakage
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    if imputation_values is None:
        imputation_values = {}
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                imputation_values[col] = median_val
                df_clean[col] = df_clean[col].fillna(median_val)
    else:
        for col, val in imputation_values.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(val)

    # 7. Log Transformation (EDA Finding: Skewness)
    if 'annual_income' in df_clean.columns:
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
    
    # Robustly identify numeric columns to scale
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Scale only continuous features, skipping binary/ordinal ones
    cols_to_scale = [
        col for col in numeric_cols 
        if X_train[col].nunique() > 2 
        and col not in ['grade_encoded', 'high_interest']
    ]
    # 1. Fit only on Training data
    scaler.fit(X_train[cols_to_scale])

    # 2. Transform Training data
    X_train_scaled = X_train.copy()
    X_train_scaled[cols_to_scale] = scaler.transform(X_train[cols_to_scale])

    # 3. Transform Validation data using the train scaler
    X_val_scaled = X_val.copy()
    X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])

    # 4. Alignment Fix
    # pd.get_dummies can create different columns if X_train has a category that X_val misses.
    # We align X_val to X_train, filling missing columns with 0.
    X_val_scaled = X_val_scaled.reindex(columns=X_train_scaled.columns, fill_value=0)

    return X_train_scaled, X_val_scaled