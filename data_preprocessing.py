import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    A modular class to handle data preprocessing steps:
    - Missing value imputation (using median for numerical, mode for categorical)
    - Categorical variable encoding
    - Feature scaling
    Prevents data leakage by fitting parameters on the training set only.
    """
    def __init__(self, target_column='Exam_Score'):
        self.target_column = target_column
        self.numerical_cols = ['Study_Hours', 'Attendance_Rate', 'Previous_Score', 'Sleep_Hours']
        self.categorical_cols = ['Extracurricular']
        
        # Fit parameters
        self.medians = {}
        self.modes = {}
        self.scaler = StandardScaler()
        
    def fit(self, X_train):
        """Fit preprocessing parameters on the training set."""
        # 1. Store medians for numerical columns
        for col in self.numerical_cols:
            if col in X_train.columns:
                self.medians[col] = X_train[col].median()
                
        # 2. Store modes for categorical columns
        for col in self.categorical_cols:
            if col in X_train.columns:
                self.modes[col] = X_train[col].mode()[0]
                
        # 3. Fit the scaler on imputed numerical data
        X_imputed = X_train.copy()
        X_imputed = self._impute(X_imputed)
        X_encoded = self._encode(X_imputed)
        
        # Fit scaler on numerical features
        self.scaler.fit(X_encoded[self.numerical_cols])
        return self
        
    def _impute(self, df):
        """Helper to impute missing values using fitted statistics."""
        df_copy = df.copy()
        for col, median_val in self.medians.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(median_val)
                
        for col, mode_val in self.modes.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(mode_val)
        return df_copy
        
    def _encode(self, df):
        """Helper to encode categorical fields."""
        df_copy = df.copy()
        for col in self.categorical_cols:
            if col in df_copy.columns:
                # Map Yes/No to 1/0
                df_copy[col] = df_copy[col].map({'Yes': 1, 'No': 0})
                # Handle unexpected values or double NaNs
                df_copy[col] = df_copy[col].fillna(0).astype(int)
        return df_copy

    def transform(self, X):
        """Apply preprocessing transformations to a dataset."""
        # 1. Impute missing values
        X_trans = self._impute(X)
        
        # 2. Encode categorical features
        X_trans = self._encode(X_trans)
        
        # 3. Scale numerical features
        X_trans[self.numerical_cols] = self.scaler.transform(X_trans[self.numerical_cols])
        
        return X_trans
        
    def fit_transform(self, X_train):
        """Fit and transform training dataset."""
        return self.fit(X_train).transform(X_train)

def load_data(filepath):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded dataset from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        raise

def prepare_splits(df, target_col='Exam_Score', test_size=0.2, random_state=42):
    """Split the raw data into Train and Test features and labels."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Split data into train (size={X_train.shape[0]}) and test (size={X_test.shape[0]}) sets.")
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Test script functionality
    df = load_data('student_scores.csv')
    X_train, X_test, y_train, y_test = prepare_splits(df)
    
    preprocessor = DataPreprocessor()
    X_train_clean = preprocessor.fit_transform(X_train)
    X_test_clean = preprocessor.transform(X_test)
    
    print("\n--- Raw Training Head ---")
    print(X_train.head(3))
    print("\n--- Preprocessed Training Head ---")
    print(X_train_clean.head(3))
    print("\nPreprocessed columns match standard scale: mean ~0, std ~1.")
    print("Preprocessors successfully compiled!")
