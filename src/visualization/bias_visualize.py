import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_bias_distribution(bias_scores_path, concept_names=None):
    # Load bias scores
    bias_scores = np.load(bias_scores_path)

    # If 2D: shape = (num_images, num_concepts), else flatten
    if bias_scores.ndim == 2:
        avg_scores = bias_scores.mean(axis=0)
        if concept_names and len(concept_names) == avg_scores.shape[0]:
            # Bar plot: average bias per concept
            plt.figure(figsize=(12, 6))
            sns.barplot(x=concept_names, y=avg_scores)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("Avg Bias Score")
            plt.title("Bias per Concept")
            plt.tight_layout()
            plt.savefig("bias_results/concept_bias_barplot.png")
            print("✅ Saved concept bias barplot to bias_results/concept_bias_barplot.png")
        else:
            print("⚠️ Concept names not provided or length mismatch — skipping barplot.")
        
        # Also plot histogram of all scores
        bias_scores = bias_scores.flatten()

    # Plot histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(bias_scores, bins=50, kde=True)
    plt.xlabel("Bias Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Bias Scores")
    plt.tight_layout()
    plt.savefig("bias_results/bias_score_distribution.png")
    print("✅ Saved bias score distribution to bias_results/bias_score_distribution.png")

if __name__ == "__main__":
    bias_scores_path = "bias_results/bias_scores.npy"
    
    # Optionally provide concept names
    concept_names = [
        "sky", "grass", "water", "tree", "rock", 
        "snow", "mountain", "branch", "sand", "fence"
    ]
    
    plot_bias_distribution(bias_scores_path, concept_names)