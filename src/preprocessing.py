import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values for numerical and categorical features."""
    df = df.copy()

    # Numeric columns: fill missing values with median
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns: fill missing values with mode
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features."""
    return pd.get_dummies(X, drop_first=True)


def prepare_data(
    df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42
):
    """Pipeline function to handle NaNs, encode features, and perform train/test split."""
    # 1. Clean missing values
    df_clean = handle_missing_values(df)

    # 2. Separate target (y) and features (X)
    y = df_clean[target_column]
    X = df_clean.drop(columns=[target_column])

    # 3. One-hot encode feature columns
    X_encoded = encode_features(X)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test