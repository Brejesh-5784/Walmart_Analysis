import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation(df, save_path="assets/correlation_heatmap.png"):
    plt.figure(figsize=(8, 8))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, fmt='.1f', cmap='Blues', square=True)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
