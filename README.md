# 🎓 Student Scorecard: Machine Learning Prediction System

A modular and interactive Machine Learning pipeline in Python that predicts student exam scores based on study hours, attendance rates, previous term performance, sleep schedules, and extracurricular participation. 

This repository includes data preprocessing pipelines (with missing value imputations and standard normalization), exploratory data analysis (EDA), model training (Linear Regression and XGBoost Regressor), and an interactive **Streamlit Web Dashboard** for live score simulation and model metrics inspection.

---

## 📂 Project Architecture

```
├── .streamlit/
│   └── config.toml                  # Streamlit theme setup (dark mode & magenta theme)
├── outputs/
│   ├── model_comparison_metrics.csv # Output table comparing regression model metrics
│   └── plots/
│       ├── correlation_matrix.png   # Pearson correlation heatmap
│       ├── study_hours_vs_score.png # Regression trend for Study Hours vs. Exam Score
│       ├── previous_score_vs_score.png # Regression trend for Previous Score vs. Exam Score
│       └── xgboost_feature_importance.png # Relative predictor weight values from XGBoost
├── app.py                           # Streamlit Interactive Dashboard entrypoint
├── data_preprocessing.py            # Preprocessor class (imputation, encoding, scaling)
├── eda.py                           # Plotting routines for correlation heatmaps and trends
├── generate_data.py                 # Synthetic dataset generator
├── main.py                          # Core Python pipeline execution script
├── requirements.txt                 # Project dependencies list
└── .gitignore                       # Git exclusion rules
```

---

## 📈 Model Performance & Evaluation

Both models were fitted on the training split and evaluated on the holdout test set (20% split) using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared ($R^2$):

| Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R-squared ($R^2$) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | **4.265%** | **5.741%** | **87.3% (0.873)** |
| **XGBoost Regressor** | 5.022% | 6.258% | 8.49% (0.849) |

### XGBoost Feature Importance
XGBoost calculated the relative importance weighting of features as:
1. **Study Hours**: `60.93%` (Most critical predictor)
2. **Previous Exam Score**: `16.09%`
3. **Attendance Rate**: `11.95%`
4. **Extracurricular Activities**: `6.33%`
5. **Sleep Schedule**: `4.70%`

---

## ⚙️ How to Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/Veer-pratap-singh/student-scorecard.git
cd student-scorecard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the ML Pipeline (Terminal output only)
To execute the data preprocessing, training, and output generation pipeline:
```bash
python main.py
```

### 4. Run the Streamlit Dashboard (Web Interface)
To launch the interactive dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🛠️ Data Preprocessing Details
- **Numerical Imputation**: Missing continuous variables are filled using training set **Medians**.
- **Categorical Imputation**: Missing values in binary columns are filled using the training set **Mode**.
- **Normalization**: Continuous features are scaled using Scikit-Learn's `StandardScaler` to prevent feature magnitude dominance.
- **Categorical Encoding**: Binary features are mapped to $1/0$ integers.
- *Strict division is maintained between training and test transforms to prevent data leakage.*
