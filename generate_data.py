import numpy as np
import pandas as pd

def generate_student_dataset(n_samples=250, random_seed=42):
    np.random.seed(random_seed)
    
    # 1. Generate study hours (1.0 to 12.0 hours)
    study_hours = np.random.uniform(1.0, 12.0, size=n_samples)
    
    # 2. Attendance rate (60% to 100%)
    attendance_rate = np.random.uniform(60.0, 100.0, size=n_samples)
    
    # 3. Previous Exam Scores (normally distributed, mean=68, std=12, clamped between 40 and 100)
    previous_score = np.random.normal(loc=68.0, scale=12.0, size=n_samples)
    previous_score = np.clip(previous_score, 40.0, 100.0)
    
    # 4. Average Sleep Hours (normally distributed, mean=7.0, std=1.2, clamped [4.0, 10.0])
    sleep_hours = np.random.normal(loc=7.0, scale=1.2, size=n_samples)
    sleep_hours = np.clip(sleep_hours, 4.0, 10.0)
    
    # 5. Extracurricular activities (Yes / No)
    extracurricular = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.45, 0.55])
    
    # 6. Define the target: Exam Score
    # Linear and interaction terms
    base_score = (
        (study_hours * 3.8) + 
        ((attendance_rate - 60) * 0.45) + 
        (previous_score * 0.42) - 
        ((8 - sleep_hours) ** 2 * 0.6) + 
        (2.5 * (extracurricular == 'Yes').astype(int))
    )
    
    # Add random normal noise (standard deviation = 4)
    noise = np.random.normal(0, 4.0, size=n_samples)
    exam_score = base_score + noise
    
    # Scale & clamp final score to percentage scale [15, 100]
    exam_score = np.clip(exam_score, 15.0, 100.0)
    
    # Round columns for cleanliness
    df = pd.DataFrame({
        'Study_Hours': np.round(study_hours, 1),
        'Attendance_Rate': np.round(attendance_rate, 1),
        'Previous_Score': np.round(previous_score, 1),
        'Sleep_Hours': np.round(sleep_hours, 1),
        'Extracurricular': extracurricular,
        'Exam_Score': np.round(exam_score, 1)
    })
    
    # Introduce random missing values (approx 5%) to represent raw data cleaning challenges
    mask_hours = np.random.rand(n_samples) < 0.05
    mask_attendance = np.random.rand(n_samples) < 0.05
    mask_extracurricular = np.random.rand(n_samples) < 0.04
    
    df.loc[mask_hours, 'Study_Hours'] = np.nan
    df.loc[mask_attendance, 'Attendance_Rate'] = np.nan
    df.loc[mask_extracurricular, 'Extracurricular'] = np.nan
    
    return df

if __name__ == '__main__':
    df = generate_student_dataset()
    df.to_csv('student_scores.csv', index=False)
    print(f"Dataset generated and saved successfully to 'student_scores.csv' ({len(df)} samples).")
    print(df.isnull().sum()) # Print missing value counts
