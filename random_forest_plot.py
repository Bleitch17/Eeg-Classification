import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import precision_score, f1_score, confusion_matrix
import sys
import os

# Import all experiment functions from random_forest.py
from random_forest import (
    experiment_initial,
    # experiment_no_rest,
    # experiment_5_channel,
    # experiment_5_channel_no_rest,
    # experiment_left_right_hand,
    # experiment_left_right_hand_5_channel,
    # experiment_left_right_hand_2_channel,
    # experiment_resting_vs_left_hand,
    # experiment_resting_vs_left_hand_2_channel,
    # experiment_resting_vs_left_hand_c3,
    # experiment_resting_vs_left_hand_c4,
    # experiment_resting_vs_left_right_hand_2_channel,
    # experiment_resting_vs_all_single_channel,
    # experiment_resting_vs_all_5_channel,
    # experiment_resting_vs_all
)

def plot_random_forest_results(experiment_name, fold_metrics, conf_matrix, class_names):
    """Plot comprehensive results from Random Forest classification"""
    accuracies, precisions, f1s = fold_metrics
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Metrics across folds
    ax1 = plt.subplot(2, 2, 1)
    folds = range(1, len(accuracies) + 1)
    
    ax1.plot(folds, accuracies, 'bo-', label='Accuracy', linewidth=2, markersize=8)
    ax1.plot(folds, precisions, 'ro-', label='Precision', linewidth=2, markersize=8)
    ax1.plot(folds, f1s, 'go-', label='F1-Score', linewidth=2, markersize=8)
    
    ax1.set_title('Metrics Across Folds')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Box plot of metrics
    ax2 = plt.subplot(2, 2, 2)
    metrics_data = pd.DataFrame({
        'Accuracy': accuracies,
        'Precision': precisions,
        'F1-Score': f1s
    })
    metrics_data.boxplot(ax=ax2)
    ax2.set_title('Distribution of Metrics')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    
    # Plot 3: Confusion Matrix
    ax3 = plt.subplot(2, 2, 3)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_title('Confusion Matrix')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')
    
    # Plot 4: Summary Statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    summary_text = f"""
    Summary Statistics:
    
    Accuracy:
    Mean: {np.mean(accuracies)*100:.2f}%
    Std:  {np.std(accuracies)*100:.2f}%
    
    Precision:
    Mean: {np.mean(precisions)*100:.2f}%
    Std:  {np.std(precisions)*100:.2f}%
    
    F1-Score:
    Mean: {np.mean(f1s)*100:.2f}%
    Std:  {np.std(f1s)*100:.2f}%
    """
    ax4.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top')
    
    plt.suptitle(f'Random Forest Results - {experiment_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    output_path = f'random_forest_{experiment_name.lower().replace(" ", "_")}_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Load your data
    flat_df = pd.read_parquet("dataset_bci_iv_2a/A01_100_90_flattened.parquet")
    
    # Define experiments
    experiments = [
        ("Initial (All Channels)", experiment_initial),
        # ("No Rest (All Channels)", experiment_no_rest),
        # ("5 Channel", experiment_5_channel),
        # ("5 Channel No Rest", experiment_5_channel_no_rest),
        # ("Left vs Right Hand", experiment_left_right_hand),
        # ("Left vs Right Hand 5 Channel", experiment_left_right_hand_5_channel),
        # ("Left vs Right Hand 2 Channel", experiment_left_right_hand_2_channel),
        # ("Resting vs Left Hand", experiment_resting_vs_left_hand),
        # ("Resting vs Left Hand 2 Channel", experiment_resting_vs_left_hand_2_channel),
        # ("Resting vs Left Hand C3", experiment_resting_vs_left_hand_c3),
        # ("Resting vs Left Hand C4", experiment_resting_vs_left_hand_c4),
        # ("Resting vs Left/Right Hand 2 Channel", experiment_resting_vs_left_right_hand_2_channel),
        # ("Resting vs All Single Channel", experiment_resting_vs_all_single_channel),
        # ("Resting vs All 5 Channel", experiment_resting_vs_all_5_channel),
        # ("Resting vs All", experiment_resting_vs_all)
    ]
    
    # Run experiments and plot results
    for experiment_name, experiment_function in experiments:
        print(f"\nRunning experiment: {experiment_name}")
        try:
            # Get metrics for this experiment
            accuracies, precisions, f1s, conf_matrix, class_report = experiment_function(flat_df)
            
            # Define class names based on the experiment
            if "Left vs Right Hand" in experiment_name:
                class_names = ['Left Hand', 'Right Hand']
            elif "No Rest" in experiment_name:
                class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
            elif "Resting vs Left Hand" in experiment_name:
                class_names = ['Rest', 'Left Hand']
            elif "Resting vs Left/Right Hand" in experiment_name:
                class_names = ['Rest', 'Left Hand', 'Right Hand']
            elif "Resting vs All" in experiment_name:
                class_names = ['Rest', 'Movement']
            else:
                class_names = ['Rest', 'Left Hand', 'Right Hand', 'Feet', 'Tongue']
            
            # Plot results
            plot_random_forest_results(
                experiment_name,
                (accuracies, precisions, f1s),
                conf_matrix,
                class_names
            )
            
            print("\nClassification Report:")
            print(class_report)
            
        except Exception as e:
            print(f"Error in experiment {experiment_name}: {str(e)}") 