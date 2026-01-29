import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import seaborn as sn

def plot_confusion_matrix(all_y_true, all_y_pred, categories=None):
    classes = np.unique(all_y_true)
    num_classes = len(classes)

    # Compute confusion matrix
    #if categories is not None:
    #    conf_matrix = confusion_matrix(all_y_true, all_y_pred, labels=categories)
    #    conf_mat_perc = confusion_matrix(all_y_true, all_y_pred, normalize='true', labels=categories)
    #else:
    conf_matrix = confusion_matrix(all_y_true, all_y_pred)
    conf_mat_perc = confusion_matrix(all_y_true, all_y_pred, normalize='true', labels=range(num_classes))
    # conf_mat_perc = conf_mat_perc*100

    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = np.unique(all_y_true)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



    #LABELS = actions  # [i for i in range(num_actions)]
    if categories is not None:
        df_cm = pd.DataFrame(conf_mat_perc, index=categories,
                            columns=categories)
    else:
        df_cm = pd.DataFrame(conf_mat_perc)

    group_counts = ["{0:0.0f}".format(value) for value in
                    conf_matrix.flatten()]
    group_percentages = ["{0:.2f}".format(value) for value in
                            conf_mat_perc.flatten()]

    labels = [f"{v1}\n{v2}" for v1, v2 in
                zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(num_classes, num_classes)
    plt.figure()
    ax = sn.heatmap(df_cm, annot=labels, fmt='', cmap='Blues', square=True, vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0)
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.ylabel('True Class', fontweight='bold')
    plt.yticks(rotation=0)
    plt.show(block=True)



