import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(
    data_path: str, 
    test_size: float = 0.2, 
    random_state: int = 2025
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads the raw Kaggle training data and splits it into local Training and 
    Validation sets using stratified sampling.

    Args:
        data_path (str): The relative or absolute path to the .csv file.
        test_size (float, optional): The proportion of the dataset to include 
            in the validation split. Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the 
            data before applying the split. Defaults to 2025.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing:
            - X_train: Training features.
            - X_val: Validation features.
            - y_train: Training target labels.
            - y_val: Validation target labels.

    Raises:
        FileNotFoundError: If the provided `data_path` does not exist.
        ValueError: If the dataset does not contain the required 'loan_paid_back' column.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find file at {data_path}")

    # Validation to ensure we are loading the correct dataset
    if 'loan_paid_back' not in df.columns:
        raise ValueError("The dataset is missing the target column 'loan_paid_back'.")
    
    # Separate Target and Features
    X = df.drop(columns=['loan_paid_back'])
    y = df['loan_paid_back']
    
    # Create Local Split
    # We use Stratified sampling to ensure the class imbalance is preserved
    # in both the training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    return X_train, X_val, y_train, y_val

def load_kaggle_test(test_path: str) -> pd.DataFrame:
    """
    Loads the production data (Kaggle Test set) which does not contain targets.

    Args:
        test_path (str): The path to the test.csv file.

    Returns:
        pd.DataFrame: The features for the submission dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        return pd.read_csv(test_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find test file at {test_path}")