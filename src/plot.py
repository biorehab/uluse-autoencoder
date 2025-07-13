import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from matplotlib import rcParams


rcParams.update({
    'font.size': 10,
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

#to plot the training history
def plot_training_history(history, class_loss=True):
    if class_loss:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.plot(history.history['reconstructed_loss'], 
                 label='Reconstruction Loss',
                 color='blue')
        ax1.plot(history.history['val_reconstructed_loss'], 
                 label='Validation Reconstruction Loss', 
                 color='lightblue', 
                 linestyle='dashed')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_title('Reconstruction Loss', fontsize=12)
        ax1.set_xlabel('Epochs', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.legend()

        combined_loss = np.array(history.history['reconstructed_loss']) + 0.1 * np.array(history.history['classifier_loss'])
        combined_val_loss = np.array(history.history['val_reconstructed_loss']) + 0.1 * np.array(history.history['val_classifier_loss'])
        
        ax2.plot(combined_loss, 
                 label='Combined Loss', 
                 color='orange')
        ax2.plot(combined_val_loss, 
                 label='Combined Validation Loss', 
                 color='red', 
                 linestyle='dashed')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_title('Combined Loss', fontsize=12)
        ax2.set_xlabel('Epochs', fontsize=10)
        ax2.set_ylabel('Loss', fontsize=10)
        ax2.legend(loc='upper right')
        fig.suptitle('Loss curves', fontsize=14)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(5,4))
        ax1.plot(history.history['loss'], 
                 label='Reconstruction Loss', 
                 color='blue')
        ax1.plot(history.history['val_loss'], 
                 label='Validation Reconstruction Loss', 
                 color='lightblue', 
                 linestyle='dashed')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_title('Reconstruction Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()
    
def plot_roc(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(fpr, tpr, color='blue',label=f'ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.tick_params(axis='both', which='major')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(conf_matrix, limb):
    num_subjects = len(conf_matrix)
    if num_subjects>5:
        rows = num_subjects//5
        fig,axs = plt.subplots(rows, 5, figsize=(25, 15))
    else:
        fig,axs = plt.subplots(1, num_subjects, figsize=(20, 10))
    axs = axs.flatten()
    for i, cm in enumerate(conf_matrix):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[i], cbar=False, annot_kws={"size": 14, "weight": "bold"})
        axs[i].set_title(f'Subject {i+1}, {limb} Limb', fontsize=14)
        axs[i].set_xlabel('Predicted', fontsize=14)
        axs[i].set_ylabel('True', fontsize=14)
        axs[i].set_xticklabels(['Non-functional', 'Functional'], fontsize=14)
        axs[i].set_yticklabels(['Non-functional', 'Functional'], fontsize=14)
        axs[i].set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plot_ensemble_loss(history_list, num_plots=None):
    
    if num_plots is None:
        num_plots = len(history_list)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 4))

    if num_plots == 1:
        axes = [axes]
    
    for i, history in enumerate(history_list):
        ax = axes[i]
        ax.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            ax.plot(history.history['val_loss'], label='Validation Loss')
        
        ax.set_title(f'AE {i+1}')
        ax.set_xlabel('Epoch')
        if i == 0:
            ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

