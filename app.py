import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from generate_data import generate_student_dataset
from data_preprocessing import load_data, prepare_splits, DataPreprocessor
from eda import run_eda_pipeline
from models import (
    train_linear_regression,
    train_xgboost,
    evaluate_model,
    plot_xgboost_feature_importance
)

# Set page configuration
st.set_page_config(
    page_title="Student Score Prediction ML Lab",
    page_icon="🎓",
    layout="wide"
)

# Premium dark theme and magenta overrides
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Orbitron:wght@500;700;900&display=swap');

    h1, h2, h3, h4 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: -0.5px;
    }
    
    .stApp {
        background-color: #050508;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Glowing header banner */
    .banner-header {
        text-align: center;
        padding: 35px 20px;
        background: linear-gradient(rgba(15, 15, 25, 0.8), rgba(5, 5, 8, 0.95)), url('banner_bg.png');
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(255, 0, 127, 0.2);
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
    }
    
    .badge {
        display: inline-block;
        background: rgba(255, 0, 127, 0.1);
        border: 1px solid #ff007f;
        color: #ff007f;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        padding: 6px 16px;
        border-radius: 30px;
        margin-bottom: 12px;
    }
    
    .main-title {
        font-size: 2.6rem;
        font-weight: 900;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #ffffff 0%, #ff007f 65%, #ff3399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .tagline {
        color: #9ea2b0;
        font-size: 1rem;
        font-weight: 300;
        max-width: 700px;
        margin: 0 auto;
    }

    /* Glassmorphic Panel styling */
    .glass-card {
        background: rgba(15, 15, 25, 0.65);
        border: 1px solid rgba(255, 0, 127, 0.15);
        border-radius: 18px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .glass-card:hover {
        border-color: rgba(255, 0, 127, 0.25);
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 15px;
        color: #ffffff;
        border-bottom: 2px solid #ff007f;
        padding-bottom: 6px;
        display: inline-block;
    }

    /* Predictions box layout */
    .prediction-output-box {
        background: linear-gradient(180deg, rgba(255, 0, 127, 0.08) 0%, rgba(0, 0, 0, 0.4) 100%);
        border: 1px dashed rgba(255, 0, 127, 0.3);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }

    .prediction-result {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.8rem;
        font-weight: 900;
        color: #ff007f;
        text-shadow: 0 0 20px rgba(255, 0, 127, 0.3);
    }
    
    .formula-box {
        background: rgba(0, 0, 0, 0.3);
        padding: 6px 15px;
        border-radius: 30px;
        display: inline-block;
        font-size: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 10px;
        font-family: 'Orbitron', sans-serif;
    }
    
    .footer-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(255, 0, 127, 0.2) 50%, transparent 100%);
        margin-top: 40px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE & INITIALIZATION -----------------
# Setup directories
output_dir = 'outputs'
plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)
data_path = 'student_scores.csv'

# Check if file exists, otherwise create it
if 'dataset_updated' not in st.session_state:
    st.session_state.dataset_updated = False

if not os.path.exists(data_path) or st.session_state.dataset_updated:
    df_raw = generate_student_dataset(n_samples=250)
    df_raw.to_csv(data_path, index=False)
    st.session_state.dataset_updated = False
else:
    df_raw = pd.read_csv(data_path)

# Fit pipeline and train models
@st.cache_resource
def execute_pipeline(df):
    # EDA execution (saves plot files)
    run_eda_pipeline(df, output_base=output_dir)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = prepare_splits(df, target_col='Exam_Score')
    
    # Preprocessor fitting
    preprocessor = DataPreprocessor()
    X_train_clean = preprocessor.fit_transform(X_train)
    X_test_clean = preprocessor.transform(X_test)
    
    # Train Models
    lr_model = train_linear_regression(X_train_clean, y_train)
    xgb_model = train_xgboost(X_train_clean, y_train)
    
    # Evaluate
    lr_metrics, _ = evaluate_model(lr_model, X_test_clean, y_test, "Linear Regression")
    xgb_metrics, _ = evaluate_model(xgb_model, X_test_clean, y_test, "XGBoost Regressor")
    
    # Save Feature Importance plots
    feature_names = list(X_train_clean.columns)
    plot_xgboost_feature_importance(xgb_model, feature_names, plots_dir)
    
    return preprocessor, lr_model, xgb_model, lr_metrics, xgb_metrics, feature_names

# Run the training logic
preprocessor, lr_model, xgb_model, lr_metrics, xgb_metrics, feature_names = execute_pipeline(df_raw)

# ----------------- HEADER BANNER -----------------
st.markdown("""
<div class="banner-header">
    <div class="badge">MODULAR MACHINE LEARNING SYSTEM</div>
    <div class="main-title">AI & ML <span class="gradient-text">Internship Lab</span></div>
    <p class="tagline">Interactive Dashboard demonstrating Student Score Predictions using Linear Regression and XGBoost.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- TABS LAYOUT -----------------
tab1, tab2, tab3 = st.tabs([
    "📂 Data & Preprocessing", 
    "📈 Model Comparison & EDA", 
    "🎯 Interactive Score Predictor"
])

# ------------- TAB 1: DATA & PREPROCESSING -------------
with tab1:
    col_d1, col_d2 = st.columns([1, 1.2], gap="large")
    
    with col_d1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">Raw Synthetic Dataset</div>
            <p style="color:#9ea2b0; font-size:0.9rem;">
                A mock student record file was generated. It includes 250 samples containing performance metrics and intentional missing rows (NaNs) to test missing value imputation.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display dataset preview
        st.dataframe(df_raw.head(8), use_container_width=True)
        
        # Check missing values
        missing_counts = df_raw.isnull().sum()
        missing_df = pd.DataFrame({'Feature': missing_counts.index, 'Missing Count': missing_counts.values})
        st.markdown("<p style='font-size:0.9rem; font-weight:600; margin-top:15px;'>Dataset Missing Value Breakdown:</p>", unsafe_allow_html=True)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
        
    with col_d2:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">Data Preprocessing Pipelines</div>
            <p style="color:#9ea2b0; font-size:0.9rem; margin-bottom:20px;">
                Before training, variables must be imputed, encoded, and scaled. We prevent data leakage by fitting parameters <b>exclusively on the training split</b>.
            </p>
            <div style="margin-bottom: 12px;">
                <b>1. Missing Value Imputation:</b>
                <ul style="color:#9ea2b0; font-size:0.85rem; padding-left:20px;">
                    <li>Numeric variables (Hours, Attendance) are imputed using training <b>Medians</b>.</li>
                    <li>Categorical variables (Extracurriculars) are imputed using training <b>Mode</b>.</li>
                </ul>
            </div>
            <div style="margin-bottom: 12px;">
                <b>2. Categorical Encoding:</b>
                <ul style="color:#9ea2b0; font-size:0.85rem; padding-left:20px;">
                    <li>Binary variables mapped to 1 (Yes) and 0 (No).</li>
                </ul>
            </div>
            <div>
                <b>3. Feature Scaling:</b>
                <ul style="color:#9ea2b0; font-size:0.85rem; padding-left:20px;">
                    <li>Scaled via standard normalization (mean = 0, std = 1) using <code>StandardScaler</code>.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display cleaned sample
        X_train, _, _, _ = prepare_splits(df_raw)
        X_clean_sample = preprocessor.transform(X_train.head(5))
        st.markdown("<p style='font-size:0.9rem; font-weight:600;'>Cleaned & Standardized Training Sample:</p>", unsafe_allow_html=True)
        st.dataframe(X_clean_sample, use_container_width=True)

# ------------- TAB 2: MODEL COMPARISON & EDA -------------
with tab2:
    col_c1, col_c2 = st.columns([1, 1.2], gap="large")
    
    with col_c1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">Model Performance Summary</div>
            <p style="color:#9ea2b0; font-size:0.9rem;">
                Both models were evaluated on the holdout test set using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display metrics table
        metrics_df = pd.DataFrame([lr_metrics, xgb_metrics])
        st.table(metrics_df)
        
        st.markdown("""
        <div class="glass-card" style="margin-top:20px;">
            <div class="section-title">XGBoost Feature Importance</div>
            <p style="color:#9ea2b0; font-size:0.9rem; margin-bottom:15px;">
                Calculated dynamically during training of XGBoost tree building. Relates each predictor's overall contribution.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display saved feature importance plot
        if os.path.exists(os.path.join(plots_dir, 'xgboost_feature_importance.png')):
            st.image(os.path.join(plots_dir, 'xgboost_feature_importance.png'), use_container_width=True)
            
    with col_c2:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">Exploratory Visualizations</div>
            <p style="color:#9ea2b0; font-size:0.9rem;">
                Interactive visualization plots compiled during our modular pipeline execution.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        vis_select = st.selectbox(
            "Select Graph Visual",
            ["Correlation Matrix Heatmap", "Study Hours vs. Target Score", "Previous Term Score vs. Target Score"]
        )
        
        if vis_select == "Correlation Matrix Heatmap":
            if os.path.exists(os.path.join(plots_dir, 'correlation_matrix.png')):
                st.image(os.path.join(plots_dir, 'correlation_matrix.png'), use_container_width=True)
        elif vis_select == "Study Hours vs. Target Score":
            if os.path.exists(os.path.join(plots_dir, 'study_hours_vs_score.png')):
                st.image(os.path.join(plots_dir, 'study_hours_vs_score.png'), use_container_width=True)
        else:
            if os.path.exists(os.path.join(plots_dir, 'previous_score_vs_score.png')):
                st.image(os.path.join(plots_dir, 'previous_score_vs_score.png'), use_container_width=True)

# ------------- TAB 3: INTERACTIVE SCORE PREDICTOR -------------
with tab3:
    col_p1, col_p2 = st.columns([0.8, 1.2], gap="large")
    
    with col_p1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title">Inference Parameters</div>
            <p style="color:#9ea2b0; font-size:0.9rem; margin-bottom:20px;">
                Adjust student behaviors below to estimate their final exam score using our trained models.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # User input widgets
        in_hours = st.slider("Daily Study Hours", 1.0, 12.0, 6.5, 0.1)
        in_attendance = st.slider("Class Attendance Rate (%)", 60.0, 100.0, 85.0, 0.5)
        in_prev_score = st.slider("Previous Score (%)", 40.0, 100.0, 70.0, 0.5)
        in_sleep = st.slider("Sleep Hours / Night", 4.0, 10.0, 7.0, 0.1)
        in_extracurricular = st.selectbox("Extracurricular Engagement", ["Yes", "No"])
        
        # Model selection
        selected_model_name = st.selectbox("Inference Model", ["Linear Regression", "XGBoost Regressor"])
        
    with col_p2:
        # Preprocess input row
        input_data = pd.DataFrame([{
            'Study_Hours': in_hours,
            'Attendance_Rate': in_attendance,
            'Previous_Score': in_prev_score,
            'Sleep_Hours': in_sleep,
            'Extracurricular': in_extracurricular
        }])
        
        # Apply scaling and transforms using fitted preprocessor
        input_clean = preprocessor.transform(input_data)
        
        # Make predictions
        if selected_model_name == "Linear Regression":
            raw_prediction = lr_model.predict(input_clean)[0]
            # Show formula details
            coefs = lr_model.coef_
            intercept_val = lr_model.intercept_
            formula_text = f"y = {coefs[0]:.2f}*Hours + {coefs[1]:.2f}*Attendance + {coefs[2]:.2f}*Prev + {coefs[3]:.2f}*Sleep + {coefs[4]:.2f}*Extra + {intercept_val:.1f}"
        else:
            raw_prediction = xgb_model.predict(input_clean)[0]
            formula_text = "Ensemble Decision Tree Formula (Non-linear)"
            
        prediction_clamped = np.clip(raw_prediction, 0.0, 100.0)
        
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding: 40px 20px;">
            <h3 style="margin-bottom: 25px; color:#ffffff;">Active Prediction Engine</h3>
            <div style="font-size:0.95rem; color:#9ea2b0; margin-bottom:5px;">PREDICTED EXAM SCORE</div>
            <div class="prediction-result">{prediction_clamped:.1f}%</div>
            <div class="formula-box">
                <span style="color:#646875;">Model:</span>
                <span style="color:#fff;">{selected_model_name}</span>
            </div>
            <div style="margin-top:15px; font-size:0.75rem; color:#646875;">
                Equation details: <code style="color:#ff3399; font-family:monospace;">{formula_text}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Draw interactive visual gauge using Plotly
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction_clamped,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Final Score Projection", 'font': {'color': '#ffffff', 'size': 12, 'family': 'Orbitron'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#646875"},
                'bar': {'color': "#ff007f"},
                'bgcolor': "rgba(13, 13, 22, 0.8)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 0, 127, 0.3)",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255, 0, 0, 0.1)'},
                    {'range': [40, 75], 'color': 'rgba(255, 255, 0, 0.1)'},
                    {'range': [75, 100], 'color': 'rgba(0, 255, 0, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 2},
                    'thickness': 0.75,
                    'value': prediction_clamped
                }
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#ffffff", 'family': "sans-serif"},
            height=200,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)

# ----------------- FOOTER -----------------
st.markdown("""
<div class="footer-divider"></div>
<div style="display: flex; flex-direction: column; align-items: center; text-align: center; padding-bottom: 20px;">
    <p style="color: #646875; font-size: 0.85rem; margin-bottom: 12px;">Student Score Prediction System • Modular AI/ML Python Pipeline</p>
</div>
""", unsafe_allow_html=True)
