import pandas as pd
import os
from data_preprocessing import load_data, prepare_splits, DataPreprocessor
from eda import run_eda_pipeline
from models import (
    train_linear_regression, 
    train_xgboost, 
    evaluate_model, 
    plot_xgboost_feature_importance
)

def run_ml_pipeline():
    print("=" * 60)
    print("  STUDENT SCORE PREDICTION SYSTEM - MACHINE LEARNING PIPELINE")
    print("=" * 60)
    
    # 1. Define Paths & Set Target
    data_path = 'student_scores.csv'
    output_dir = 'outputs'
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check if dataset exists, otherwise generate it
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Running generator script...")
        from generate_data import generate_student_dataset
        df = generate_student_dataset()
        df.to_csv(data_path, index=False)
        print("Dataset generated successfully.")
        
    # Load dataset
    df = load_data(data_path)
    
    # 2. Exploratory Data Analysis
    print("\n>>> Starting Exploratory Data Analysis (EDA)...")
    run_eda_pipeline(df, output_base=output_dir)
    print("EDA completed. Visualizations generated.")
    
    # 3. Data Split
    print("\n>>> Preparing Train/Test splits...")
    X_train, X_test, y_train, y_test = prepare_splits(df, target_col='Exam_Score')
    
    # 4. Data Preprocessing (Impute, Encode, Scale)
    print("\n>>> Applying Data Preprocessing Pipeline...")
    preprocessor = DataPreprocessor()
    
    # Fit on training split and transform training features
    X_train_clean = preprocessor.fit_transform(X_train)
    # Transform testing features using parameters fitted on training split
    X_test_clean = preprocessor.transform(X_test)
    
    feature_names = list(X_train_clean.columns)
    print(f"Cleaned Feature variables: {feature_names}")
    
    # 5. Model Training & Evaluation
    print("\n>>> Training Models...")
    
    # Baseline Model: Linear Regression
    lr_model = train_linear_regression(X_train_clean, y_train)
    lr_metrics, lr_preds = evaluate_model(lr_model, X_test_clean, y_test, "Linear Regression")
    
    # Advanced Model: XGBoost Regressor
    xgb_model = train_xgboost(X_train_clean, y_train)
    xgb_metrics, xgb_preds = evaluate_model(xgb_model, X_test_clean, y_test, "XGBoost Regressor")
    
    # 6. Model Comparison
    print("\n" + "=" * 50)
    print("                MODEL PERFORMANCE COMPARISON")
    print("=" * 50)
    comparison_df = pd.DataFrame([lr_metrics, xgb_metrics])
    print(comparison_df.to_string(index=False))
    print("=" * 50)
    
    # Save comparison dataframe
    comparison_path = os.path.join(output_dir, 'model_comparison_metrics.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved model comparison table to {comparison_path}")
    
    # 7. XGBoost Feature Importance
    print("\n>>> Generating Feature Importance visualization...")
    feat_imp = plot_xgboost_feature_importance(xgb_model, feature_names, plots_dir)
    
    print("\nRelative Feature Importances:")
    for idx, row in feat_imp.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")
        
    print("\n" + "=" * 60)
    print("  PIPELINE EXECUTION FINISHED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    run_ml_pipeline()
