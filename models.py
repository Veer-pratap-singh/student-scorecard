import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

def train_linear_regression(X_train, y_train):
    """Train a baseline Linear Regression model."""
    print("Training Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

def train_xgboost(X_train, y_train, random_state=42):
    """Train an advanced XGBoost Regressor model."""
    print("Training XGBoost Regressor model...")
    # Standard hyperparameters for regression
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        random_state=random_state,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model, X_test, y_test, model_name):
    """
    Calculate and return performance evaluation metrics:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R² Score (Coefficient of Determination)
    """
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        'Model': model_name,
        'MAE': round(float(mae), 3),
        'RMSE': round(float(rmse), 3),
        'R2': round(float(r2), 3)
    }
    
    print(f"\nEvaluation Results for {model_name}:")
    print(f"  MAE : {metrics['MAE']:.3f}")
    print(f"  RMSE: {metrics['RMSE']:.3f}")
    print(f"  R²  : {metrics['R2']:.3f}")
    
    return metrics, predictions

def plot_xgboost_feature_importance(xgb_model, feature_names, save_dir):
    """
    Plot and save feature importances calculated by the XGBoost model.
    Highlights which features have the highest predictive weight.
    """
    # Extract importances
    importances = xgb_model.feature_importances_
    
    # Create DataFrame
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Configure plotting style
    plt.figure(figsize=(8, 5))
    
    # Plot using a custom magenta palette
    sns.barplot(
        data=feat_imp,
        x='Importance',
        y='Feature',
        hue='Feature',
        palette=sns.color_palette("light:#ff007f_r", n_colors=len(feature_names)),
        legend=False
    )
    
    plt.title("Feature Importance - XGBoost Regressor", fontsize=12, fontweight='bold', pad=12, color='#ff007f')
    plt.xlabel("Relative Importance Score", fontsize=10)
    plt.ylabel("Predictor Features", fontsize=10)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'xgboost_feature_importance.png')
    plt.savefig(save_path, facecolor='#050508', dpi=150)
    plt.close()
    
    print(f"Models: Saved feature importance plot to {save_path}")
    return feat_imp
