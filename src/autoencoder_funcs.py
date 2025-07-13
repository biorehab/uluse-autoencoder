import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, confusion_matrix, classification_report, auc
import matplotlib.pyplot as plt

#for early stopping and learning rate reduction
def get_callbacks():
    return [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                             patience=10, 
                                             restore_best_weights=True, 
                                             verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=0.3, 
                                                 patience=5, 
                                                 min_lr=1e-7, 
                                                 verbose=0),
           ]

#determine reconstruction error threshold for approach 2
def calculate_threshold(reconstruction_errors, labels):
    fpr, tpr, thresholds = roc_curve(labels, 
                                     reconstruction_errors, 
                                     pos_label = 0)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

#compute mse for reconstructed and original data
def compute_mse(original, reconstructed):
    errors = []
    for i in range(original.shape[0]):
        error = tf.reduce_mean(tf.keras.losses.mean_squared_error(original[i], reconstructed[i]))
        errors.append(error.numpy())
    return np.array(errors)

#compute mae for reconstructed and original data
def compute_mae(original, reconstructed):
    errors = []
    for i in range(original.shape[0]):
        error = tf.reduce_mean(tf.keras.losses.mean_absolute_error(original[i], reconstructed[i]))
        errors.append(error.numpy())
    return np.array(errors)

#compute msle for reconstructed and original data
def compute_msle(original, reconstructed):
    errors = []
    for i in range(original.shape[0]):
        original_flat = tf.reshape(original[i], [-1])
        reconstructed_flat = tf.reshape(reconstructed[i], [-1])
        error = tf.reduce_mean(tf.keras.losses.mean_squared_logarithmic_error(original_flat, reconstructed_flat))
        errors.append(error.numpy())  
    return np.array(errors)

#train model with combined loss
def train_model_with_class_loss(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    return model.fit(X_train, {'reconstructed': X_train, 'classifier': y_train},
                     validation_data=(X_test, {'reconstructed': X_test, 'classifier': y_test}),
                     epochs=epochs, 
                     batch_size=batch_size, 
                     verbose=0,
                     callbacks=get_callbacks())

#train model without class loss
def train_model_without_class_loss(model, X_train, X_test, epochs, batch_size):
    return model.fit(X_train, X_train,
                     validation_data=(X_test, X_test),
                     epochs=epochs, 
                     batch_size=batch_size, 
                     verbose=0,
                     callbacks=get_callbacks())



