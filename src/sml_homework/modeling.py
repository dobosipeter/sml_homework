import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def create_model(
    model_type: str = "random_forest", 
    params: dict[str, any] | None = None
) -> BaseEstimator:
    """
    Factory function to initialize a model with the correct random state.
    
    Args:
        model_type (str): 'random_forest', 'xgboost', or 'logistic'.
        params (dict | None): Dictionary of hyperparameters for the model.

    Returns:
        BaseEstimator: The initialized Scikit-Learn (or compatible) estimator.
    """
    if params is None:
        params = {}

    # Ensure reproducibility
    random_state = 2025

    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=random_state, **params)
    elif model_type == "xgboost":
        # eval_metric='logloss' is standard for binary classification
        model = XGBClassifier(
            random_state=random_state, 
            eval_metric='logloss', 
            **params
        )
    elif model_type == "logistic":
        model = LogisticRegression(random_state=random_state, **params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return model

def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    model_type: str = "random_forest",
    params: dict[str, any] | None = None
) -> BaseEstimator:
    """
    Creates and trains a model on the provided training data.

    Args:
        X_train (pd.DataFrame): Training features (already scaled).
        y_train (pd.Series): Training targets.
        model_type (str): The type of model to initialize.
        params (dict | None): Hyperparameters.

    Returns:
        BaseEstimator: The trained model.
    """
    model = create_model(model_type, params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(
    model: BaseEstimator, 
    X_val: pd.DataFrame, 
    y_val: pd.Series
) -> float:
    """
    Evaluates the model using ROC AUC Score.
    
    Uses 'predict_proba' instead of 'predict' because ROC AUC requires 
    probability scores to evaluate performance across different thresholds.

    Args:
        model (BaseEstimator): The trained model.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): True validation targets.

    Returns:
        float: The ROC AUC score.
    """
    # predict_proba returns array of shape (n_samples, 2)
    # Col 0 = Prob of Class 0 (Default)
    # Col 1 = Prob of Class 1 (Paid Back)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    score = roc_auc_score(y_val, y_pred_proba)
    return score

def generate_submission(
    model: BaseEstimator, 
    X_test: pd.DataFrame, 
    test_ids: pd.Series
) -> pd.DataFrame:
    """
    Generates the submission DataFrame expected by Kaggle.

    Args:
        model (BaseEstimator): The trained model.
        X_test (pd.DataFrame): The test set features.
        test_ids (pd.Series): The ID column corresponding to X_test.

    Returns:
        pd.DataFrame: A dataframe with 'id' and 'loan_paid_back' probabilities.
    """
    # Predict probabilities for the positive class (1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    submission = pd.DataFrame({
        'id': test_ids,
        'loan_paid_back': y_pred_proba
    })
    
    return submission