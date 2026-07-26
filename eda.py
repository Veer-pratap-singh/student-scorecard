import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set visual styles for premium tech look
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.facecolor': '#050508',
    'axes.facecolor': '#0d0d16',
    'text.color': '#ffffff',
    'axes.labelcolor': '#ffffff',
    'xtick.color': '#9ea2b0',
    'ytick.color': '#9ea2b0',
    'grid.color': '#22222a',
    'font.family': 'sans-serif'
})

def create_output_dirs(base_dir='outputs'):
    """Create directory structure for outputs if it does not exist."""
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def plot_correlations(df, save_dir):
    """
    Generate and save a correlation heatmap of the student metrics.
    Categorical fields are temporarily mapped to numeric for complete correlation assessment.
    """
    df_corr = df.copy()
    if 'Extracurricular' in df_corr.columns:
        df_corr['Extracurricular'] = df_corr['Extracurricular'].map({'Yes': 1, 'No': 0})
        
    # Calculate correlations
    corr_matrix = df_corr.corr()
    
    plt.figure(figsize=(8, 6))
    
    # Custom magenta/dark color palette
    cmap = sns.diverging_palette(220, 340, s=90, l=50, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .8},
        annot_kws={"size": 10}
    )
    
    plt.title("Correlation Heatmap: Student Metrics & Score", fontsize=14, fontweight='bold', pad=15, color='#ff007f')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'correlation_matrix.png')
    plt.savefig(save_path, facecolor='#050508', dpi=150)
    plt.close()
    print(f"EDA: Saved correlation heatmap to {save_path}")

def plot_feature_relationships(df, save_dir):
    """Plot key scatter plots showing relationship between predictors and final exam score."""
    # Plot Study Hours vs. Exam Score
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df, 
        x='Study_Hours', 
        y='Exam_Score', 
        hue='Extracurricular',
        palette={'Yes': '#ff007f', 'No': '#9ea2b0'}, 
        alpha=0.8, 
        edgecolor='w', 
        s=60
    )
    plt.title("Exam Score vs. Daily Study Hours", fontsize=12, fontweight='bold', pad=12, color='#ff007f')
    plt.xlabel("Daily Study Hours", fontsize=10)
    plt.ylabel("Exam Score (%)", fontsize=10)
    plt.legend(title='Extracurricular', facecolor='#0d0d16', edgecolor='none')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'study_hours_vs_score.png')
    plt.savefig(save_path, facecolor='#050508', dpi=150)
    plt.close()
    
    # Plot Previous Score vs Exam Score
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=df, 
        x='Previous_Score', 
        y='Exam_Score',
        scatter_kws={'alpha':0.6, 'color': '#ff3399', 's':35},
        line_kws={'color': '#ff007f', 'linewidth': 2}
    )
    plt.title("Exam Score Trend vs. Previous Score", fontsize=12, fontweight='bold', pad=12, color='#ff007f')
    plt.xlabel("Previous Term Score (%)", fontsize=10)
    plt.ylabel("Final Exam Score (%)", fontsize=10)
    plt.tight_layout()
    
    save_path2 = os.path.join(save_dir, 'previous_score_vs_score.png')
    plt.savefig(save_path2, facecolor='#050508', dpi=150)
    plt.close()
    
    print(f"EDA: Saved scatter plots to {save_dir}")

def run_eda_pipeline(df, output_base='outputs'):
    """Orchestrate EDA steps."""
    plots_dir = create_output_dirs(output_base)
    # Fill missing values temporarily for plotting clarity
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include='number').columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    df_clean['Extracurricular'] = df_clean['Extracurricular'].fillna(df_clean['Extracurricular'].mode()[0])
    
    plot_correlations(df_clean, plots_dir)
    plot_feature_relationships(df_clean, plots_dir)
