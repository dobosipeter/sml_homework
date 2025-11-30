import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_target_distribution(
    y: pd.Series, 
    title: str = "Target Class Distribution"
) -> None:
    """
    Visualizes the distribution of the target variable to highlight class imbalance.
    Adds percentage labels to the bars for clarity.

    Args:
        y (pd.Series): The target variable series (e.g., loan_paid_back).
        title (str, optional): Title for the chart. Defaults to "Target Class Distribution".
    """
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=y)
    plt.title(title)
    plt.xlabel("Target Class")
    plt.ylabel("Count")
    
    # Add percentage labels to bars
    total = len(y)
    for p in ax.patches:
        height = p.get_height()
        percentage = f'{100 * height / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y_coord = height
        ax.annotate(percentage, (x, y_coord), ha='center', va='bottom')
        
    plt.show()

def plot_numerical_distributions(
    df: pd.DataFrame, 
    columns: list[str] | None = None
) -> None:
    """
    Plots histograms with Kernel Density Estimate for numerical features 
    to analyze distributions and skewness.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list[str] | None, optional): Specific columns to plot. 
            If None, selects all numeric columns except 'id' and 'loan_paid_back'.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
        # Remove target or ID if present, as they don't need distribution plots
        columns = [c for c in columns if c not in ['id', 'loan_paid_back']]

    num_cols = len(columns)
    if num_cols == 0:
        return

    # Calculate grid size (3 plots per row)
    rows = math.ceil(num_cols / 3)
    
    plt.figure(figsize=(15, 4 * rows))
    
    for i, col in enumerate(columns):
        plt.subplot(rows, 3, i + 1)
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(
    df: pd.DataFrame, 
    columns: list[str] | None = None
) -> None:
    """
    Plots count plots for categorical features to understand frequency and cardinality.
    Automatically switches to horizontal bars if cardinality is high.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list[str] | None, optional): Specific columns to plot.
            If None, selects all object/category columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    num_cols = len(columns)
    if num_cols == 0:
        return

    # Calculate grid size (2 plots per row for categorical to give more width)
    rows = math.ceil(num_cols / 2)
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, col in enumerate(columns):
        plt.subplot(rows, 2, i + 1)
        
        # Check cardinality to decide on rotation or orientation
        if df[col].nunique() > 10:
            sns.countplot(y=df[col], order=df[col].value_counts().index)
            plt.xlabel('Count')
            plt.ylabel(col)
        else:
            sns.countplot(x=df[col], order=df[col].value_counts().index)
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
        plt.title(f'Distribution of {col}')
        
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(
    df: pd.DataFrame, 
    method: str = 'pearson'
) -> None:
    """
    Plots a heatmap of correlations between numerical features.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        method (str, optional): Correlation method ('pearson', 'spearman', 'kendall'). 
                                Use 'spearman' for ordinal data. Defaults to 'pearson'.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Drop ID if present
    if 'id' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['id'])
        
    plt.figure(figsize=(10, 8))
    corr = numeric_df.corr(method=method)
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        mask=mask,
        cbar=True,
        square=True
    )
    plt.title(f"Feature Correlation Matrix ({method.capitalize()})")
    plt.show()

def plot_boxplots(
    df: pd.DataFrame, 
    columns: list[str] | None = None
) -> None:
    """
    Plots boxplots for numerical features to visually identify outliers.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        columns (list[str] | None, optional): Specific columns to plot.
            If None, selects all numeric columns except 'id' and target.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
        columns = [c for c in columns if c not in ['id', 'loan_paid_back']]

    num_cols = len(columns)
    if num_cols == 0:
        return

    rows = math.ceil(num_cols / 3)
    
    plt.figure(figsize=(15, 4 * rows))
    
    for i, col in enumerate(columns):
        plt.subplot(rows, 3, i + 1)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        
    plt.tight_layout()
    plt.show()

def plot_scatter(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    hue: str | None = None
) -> None:
    """
    Plots a scatter plot to visualize relationships between two continuous variables.

    Args:
        df (pd.DataFrame): The dataframe.
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis.
        hue (str | None, optional): Column name for color coding (e.g., target class).
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, alpha=0.6)
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    plt.show()