import seaborn as sns
import matplotlib.pyplot as plt

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
        ax1.set_title('Reconstruction Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(history.history['classifier_loss'], 
                 label='Classification Loss', 
                 color='orange')
        ax2.plot(history.history['val_classifier_loss'], 
                 label='Validation Classification Loss', 
                 color='red', 
                 linestyle='dashed')
        ax2.set_title('Classification Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        ax1.plot(history.history['loss'], 
                 label='Reconstruction Loss', 
                 color='blue')
        ax1.plot(history.history['val_loss'], 
                 label='Validation Reconstruction Loss', 
                 color='lightblue', 
                 linestyle='dashed')
        ax1.set_title('Reconstruction Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
    
    plt.tight_layout()
    plt.show()