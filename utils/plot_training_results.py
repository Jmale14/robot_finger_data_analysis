import matplotlib.pyplot as plt
import numpy as np


# Plot average training history across folds
def plot_average_history(histories, metric):
    average_metric = np.mean([history.history[metric] for history in histories], axis=0)
    plt.plot(average_metric, label=f'Average {metric}')

def plot_training_results(fold_histories):
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 12))

    # Plot loss
    plt.subplot(2, 2, 1)
    plot_average_history(fold_histories, 'loss')
    plot_average_history(fold_histories, 'val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plot_average_history(fold_histories, 'accuracy')
    plot_average_history(fold_histories, 'val_accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot f1
    plt.subplot(2, 2, 3)
    plot_average_history(fold_histories, 'f1_score')
    plot_average_history(fold_histories, 'val_f1_score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # Plot prec and rec
    plt.subplot(2, 2, 4)
    plot_average_history(fold_histories, 'precision')
    plot_average_history(fold_histories, 'recall')
    plot_average_history(fold_histories, 'val_precision')
    plot_average_history(fold_histories, 'val_recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Show plots
    plt.show()