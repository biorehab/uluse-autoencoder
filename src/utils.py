import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
from scipy import stats
from autoencoder import build_autoencoder
from autoencoder_funcs import train_model_with_class_loss, train_model_without_class_loss, calculate_threshold, compute_mae, compute_mse
from plot import plot_training_history
from metrics import youden_index

#For approach 1, random forest on latent space
def run_rf(encoder, X_train, X_test, y_train):
    latent_test = encoder.predict(X_test, verbose=0)
    latent_train = encoder.predict(X_train, verbose=0)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(latent_train, y_train.ravel())
    y_pred = clf.predict(latent_test)
    return y_pred

#For approach 2, 3a, 3b, get the func to calculate error
def get_error_function(loss_fn):
    
    loss_to_error = {
        tf.keras.losses.mean_squared_error: compute_mse,
        tf.keras.losses.MeanSquaredError: compute_mse,
        tf.keras.losses.mean_absolute_error: compute_mae,
        tf.keras.losses.MeanAbsoluteError: compute_mae
    }
    
    if isinstance(loss_fn, type):
        loss_fn = loss_fn()
    
    return loss_to_error.get(loss_fn, compute_mse)

#for autoencoder, get the initializer
def get_initializer(init_name, seed=42):
    initializers = {
        'he_normal': tf.keras.initializers.HeNormal(seed=seed),
        'he_uniform': tf.keras.initializers.HeUniform(seed=seed),
        'glorot_normal': tf.keras.initializers.GlorotNormal(seed=seed),
        'glorot_uniform': tf.keras.initializers.GlorotUniform(seed=seed),
        'random_normal': tf.keras.initializers.RandomNormal(seed=seed),
        'random_uniform': tf.keras.initializers.RandomUniform(seed=seed),
        'zeros': tf.keras.initializers.Zeros()
    }
    return initializers.get(init_name)

#for autoencoder, get the loss func
def get_loss_function(loss_name):
    losses = {
        'mse': tf.keras.losses.mean_squared_error,
    }
    return losses.get(loss_name)

#for train-test evaluation after finding best mean val score: process fold for approach 1 and 2
def process_fold(X_train, X_test, y_train, y_test, subjects, limbs, train_idx, test_idx, window_size, 
                 architecture, hp, epochs, batch_size, class_loss, plot_hist, approach, error_func):
    
    
    X_train, y_train, subject_ids_train, limb_ids_train = create_sliding_windows(X_train, 
                                                                                 y_train, 
                                                                                 subjects[train_idx], 
                                                                                 limbs[train_idx], 
                                                                                 None, window_size)
    X_test, y_test, subject_ids_test, limb_ids_test = create_sliding_windows(X_test, 
                                                                             y_test, 
                                                                             subjects[test_idx], 
                                                                             limbs[test_idx], 
                                                                             None, window_size )
    model, encoder = build_autoencoder(architecture, hp, 
                                       input_shape=(X_train.shape[1], 
                                                    X_train.shape[2]),
                                       num_classes = 1, 
                                       class_loss = class_loss)

    if class_loss:
        history = train_model_with_class_loss(model, 
                                              X_train, 
                                              y_train, 
                                              X_test, 
                                              y_test, 
                                              epochs, 
                                              batch_size)
    else:
        history = train_model_without_class_loss(model, 
                                                 X_train, 
                                                 X_test,
                                                 epochs, 
                                                 batch_size)

    # if plot_hist:
    #     plot_training_history(history)

    error_func = get_error_function(hp['loss'])
    if approach == 1:
        y_pred = run_rf(encoder, 
                        X_train, 
                        X_test, 
                        y_train)
    elif approach == 2:
        recon_train = error_func(X_train, 
                                 model.predict(X_train, verbose=0))
        recon_test = error_func(X_test, 
                                model.predict(X_test, verbose=0))
        threshold = calculate_threshold(recon_train, 
                                        y_train)
        y_pred = (recon_test <= threshold).astype(int)
    return y_test, y_pred, subject_ids_test, limb_ids_test

#for train-test evaluation after finding best mean val score: process fold for approach 3a
def process_fold3a(X_train, X_test, y_train, y_test, subjects, limbs, train_idx, test_idx, 
                window_size, architecture, hp, epochs, batch_size, 
                class_loss, plot_hist, approach, error_func=None):
    
    X_train, y_train, subject_ids_train, limb_ids_train = create_sliding_windows(X_train, 
                                                                                 y_train, 
                                                                                 subjects[train_idx], 
                                                                                 limbs[train_idx], 
                                                                                 None, window_size)
    X_test, y_test, subject_ids_test, limb_ids_test = create_sliding_windows(X_test, 
                                                                             y_test, 
                                                                             subjects[test_idx], 
                                                                             limbs[test_idx], 
                                                                             None, window_size)
    X_train_class_0 = X_train[y_train == 0]
    X_train_class_1 = X_train[y_train == 1]
    
    model0, encoder0 = build_autoencoder(architecture, hp, 
                                         input_shape=(X_train_class_0.shape[1], 
                                                      X_train_class_0.shape[2]), 
                                         num_classes = 1, 
                                         class_loss = class_loss)
    
    model1, encoder1 = build_autoencoder(architecture, hp, 
                                         input_shape=(X_train_class_1.shape[1], 
                                                      X_train_class_1.shape[2]), 
                                         num_classes = 1, 
                                         class_loss = class_loss)
    
    if class_loss: 
        history0 = train_model_with_class_loss(model0, 
                                               X_train_class_0, 
                                               np.zeros((len(X_train_class_0))), 
                                               X_test, 
                                               y_test, 
                                               epochs, 
                                               batch_size)
        history1 = train_model_with_class_loss(model1, 
                                               X_train_class_1, 
                                               np.ones((len(X_train_class_1))), 
                                               X_test, 
                                               y_test, 
                                               epochs, 
                                               batch_size)
        recon_error_0 = compute_mae(X_test, 
                                    model0.predict(X_test, verbose=0)[0])
        recon_error_1 = compute_mae(X_test, 
                                    model1.predict(X_test, verbose=0)[0])
    else: 
        history0 = train_model_without_class_loss(model0, 
                                                  X_train_class_0,
                                                  X_test, 
                                                  epochs, 
                                                  batch_size)
        history1 = train_model_without_class_loss(model1, 
                                                  X_train_class_1, 
                                                  X_test, 
                                                  epochs, 
                                                  batch_size)
        recon_error_0 = compute_mae(X_test, 
                                    model0.predict(X_test, verbose=0))
        recon_error_1 = compute_mae(X_test, 
                                    model1.predict(X_test, verbose=0))
    if plot_hist: 
        plot_training_history(history0)
        plot_training_history(history1)

    y_pred = (recon_error_1 < recon_error_0).astype(int)
    return y_test, y_pred, subject_ids_test, limb_ids_test

#for train-test evaluation after finding best mean val score: process fold for approach 3b
def process_fold3b(X_train, X_test, y_train, y_test, subjects, limbs, usetypes, train_idx, test_idx, 
                window_size, architecture, hp, epochs, batch_size, class_loss, plot_hist, error_func):
    
    
    X_train, y_train, subject_ids_train, limb_ids_train, use_ids_train = create_sliding_windows(X_train, 
                                                                                                y_train, 
                                                                                                subjects[train_idx], 
                                                                                                limbs[train_idx], 
                                                                                                usetypes[train_idx], 
                                                                                                window_size)
    X_test, y_test, subject_ids_test, limb_ids_test, use_ids_test = create_sliding_windows(X_test, 
                                                                                           y_test, 
                                                                                           subjects[test_idx], 
                                                                                           limbs[test_idx], 
                                                                                           usetypes[test_idx], 
                                                                                           window_size)

    autoencoders = {}
    history_list = {}
    recon_errors = []

    if class_loss: 
        for use_type_val in [1, 2, 3]:
            X_train_use = X_train[use_ids_train == use_type_val]
            y_train_use = y_train[use_ids_train == use_type_val]
            autoencoder, encoder = build_autoencoder(architecture, hp, 
                                                     input_shape=(X_train_use.shape[1], 
                                                                  X_train_use.shape[2]), 
                                                     num_classes = 1, 
                                                     class_loss = class_loss)
            history = train_model_with_class_loss(autoencoder, 
                                                  X_train_use, 
                                                  y_train_use, 
                                                  X_test,
                                                  y_test, 
                                                  epochs, 
                                                  batch_size)
            history_list[use_type_val] = history
            autoencoders[use_type_val] = autoencoder
        for use_type_val, autoencoder in autoencoders.items():
            recon_error = compute_mae(X_test, 
                                      autoencoder.predict(X_test, verbose=0)[0])
            recon_errors.append(recon_error)
    else:
        for use_type_val in [1, 2, 3]:
            X_train_use = X_train[use_ids_train == use_type_val]
            y_train_use = y_train[use_ids_train == use_type_val]
            autoencoder, encoder = build_autoencoder(architecture, hp, 
                                                     input_shape=(X_train_use.shape[1], 
                                                                  X_train_use.shape[2]), 
                                                     num_classes = 1, 
                                                     class_loss = class_loss)
            history = train_model_without_class_loss(autoencoder, 
                                                     X_train_use, 
                                                     X_test, 
                                                     epochs, 
                                                     batch_size)
            history_list[use_type_val] = history
            autoencoders[use_type_val] = autoencoder
        for use_type_val, autoencoder in autoencoders.items():
            recon_error = compute_mae(X_test, 
                                      autoencoder.predict(X_test, verbose=0))
            recon_errors.append(recon_error)

    recon_errors = np.array(recon_errors)
    min_error_idx = np.argmin(recon_errors, axis=0)
    y_pred = (min_error_idx != 2).astype(int) 
    return y_test, y_pred, subject_ids_test, limb_ids_test


#function for nested cross validation for approach 1
def run_nested_cv(window_size, X, y, subjects, limbs, architecture, hp_grid, epochs, batch_size,
                class_loss, plot_hist, approach, error_func=None):
    logo_outer = LeaveOneGroupOut()
    all_results = []

    for outer_fold, (train_val_idx, test_idx) in enumerate(logo_outer.split(X, groups=subjects)):
        tf.keras.backend.clear_session()
        X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
        y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
        subjects_train_val = subjects.iloc[train_val_idx]
        limbs_train_val = limbs.iloc[train_val_idx]

        best_score = -np.inf
        best_hp = None

        for hp_set in ParameterGrid(hp_grid):
            current_score = validate_hyperparameters(X_train_val, 
                                                     y_train_val, 
                                                     subjects_train_val, 
                                                     limbs_train_val,
                                                     window_size, 
                                                     architecture, 
                                                     hp_set, 
                                                     epochs, 
                                                     batch_size,
                                                     class_loss, 
                                                     approach, 
                                                     error_func)

            if current_score > best_score:
                best_score = current_score
                best_hp = hp_set
                print(f"New best HP found: {hp_set} with score: {current_score:.3f}")

        y_true, y_pred, test_subjects, test_limbs = process_fold(X_train_val, 
                                                                 X_test, 
                                                                 y_train_val, 
                                                                 y_test,
                                                                 subjects, 
                                                                 limbs, 
                                                                 train_val_idx, 
                                                                 test_idx,
                                                                 window_size, 
                                                                 architecture, 
                                                                 best_hp, 
                                                                 epochs, 
                                                                 batch_size,
                                                                 class_loss, 
                                                                 plot_hist, 
                                                                 approach, 
                                                                 error_func)

        for subject in np.unique(test_subjects):
            for limb in np.unique(test_limbs):
                mask = (test_subjects == subject) & (test_limbs == limb)
                if np.sum(mask) > 0:
                    sens, spec, yi = youden_index(y_true[mask], y_pred[mask])
                    print(f"Limb: {limb}",
                          f"Subject: {subject}",
                          f"Youden Index: {yi}",
                          f"Sensitivity: {sens}",
                          f"Specificity: {spec}")
                    all_results.append({
                        'subject': subject,
                        'limb': limb,
                        'youden_index': yi,
                        'sensitivity': sens,
                        'specificity': spec,
                        'hyperparameters': best_hp
                    })

    return pd.DataFrame(all_results)


#function for nested cross validation for approach 3a
def run_nested_cv_3a(window_size, X, y, subjects, limbs, architecture, hp_grid, epochs, batch_size, class_loss, plot_hist, error_func=None):
    logo_outer = LeaveOneGroupOut()
    all_results = []

    for outer_fold, (train_val_idx, test_idx) in enumerate(logo_outer.split(X, groups=subjects)):
        tf.keras.backend.clear_session()
        X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
        y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
        subjects_train_val = subjects.iloc[train_val_idx]
        limbs_train_val = limbs.iloc[train_val_idx]

        best_score = -np.inf
        best_hp = None

        for hp_set in ParameterGrid(hp_grid):
            current_score = validate_hyperparameters_3a(X_train_val, 
                                                        y_train_val, 
                                                        subjects_train_val, 
                                                        limbs_train_val,
                                                        window_size, 
                                                        architecture,
                                                        hp_set, 
                                                        epochs, 
                                                        batch_size,
                                                        class_loss, 
                                                        error_func)

            if current_score > best_score:
                best_score = current_score
                best_hp = hp_set
                print(f"New best HP found: {hp_set} with score: {current_score:.3f}")

        y_true, y_pred, test_subjects, test_limbs = process_fold3a(X_train_val, 
                                                                   X_test, 
                                                                   y_train_val, 
                                                                   y_test, 
                                                                   subjects, 
                                                                   limbs, 
                                                                   train_val_idx, 
                                                                   test_idx, 
                                                                   window_size, 
                                                                   architecture, 
                                                                   best_hp, 
                                                                   epochs, 
                                                                   batch_size, 
                                                                   class_loss, 
                                                                   plot_hist, 
                                                                   error_func)

        for subject in np.unique(test_subjects):
            for limb in np.unique(test_limbs):
                mask = (test_subjects == subject) & (test_limbs == limb)
                if np.sum(mask) > 0:
                    sens, spec, yi = youden_index(y_true[mask], y_pred[mask])
                    print(f"Limb: {limb}",
                          f"Subject: {subject}",
                          f"Youden Index: {yi}",
                          f"Sensitivity: {sens}",
                          f"Specificity: {spec}")
                    all_results.append({
                        'subject': subject,
                        'limb': limb,
                        'youden_index': yi,
                        'sensitivity': sens,
                        'specificity': spec,
                        'hyperparameters': best_hp
                    })

    return pd.DataFrame(all_results)

#function for nested cross validation for approach 3b
def run_nested_cv_3b(window_size, X, y, subjects, limbs, usetypes, architecture, hp_grid, epochs, batch_size,
                    class_loss, plot_hist, error_func=None):
    logo_outer = LeaveOneGroupOut()
    all_results = []

    for outer_fold, (train_val_idx, test_idx) in enumerate(logo_outer.split(X, groups=subjects)):
        tf.keras.backend.clear_session()
        X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
        y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
        subjects_train_val = subjects.iloc[train_val_idx]
        limbs_train_val = limbs.iloc[train_val_idx]
        usetypes_train_val = usetypes.iloc[train_val_idx]
        usetypes_test = usetypes.iloc[test_idx]

        best_score = -np.inf
        best_hp = None

        for hp_set in ParameterGrid(hp_grid):
            current_score = validate_hyperparameters_3b(X_train_val, 
                                                        y_train_val, 
                                                        subjects_train_val, 
                                                        limbs_train_val, 
                                                        usetypes_train_val, window_size, 
                                                        architecture, 
                                                        hp_set, 
                                                        epochs, 
                                                        batch_size, class_loss,
                                                        error_func)

            if current_score > best_score:
                best_score = current_score
                best_hp = hp_set
                print(f"New best HP found: {hp_set} with score: {current_score:.3f}")

        y_true, y_pred, test_subjects, test_limbs = process_fold3b(X_train_val, 
                                                                   X_test, 
                                                                   y_train_val, 
                                                                   y_test, subjects, 
                                                                   limbs, 
                                                                   usetypes, 
                                                                   train_val_idx, 
                                                                   test_idx, 
                                                                   window_size, 
                                                                   architecture, 
                                                                   best_hp, 
                                                                   epochs, 
                                                                   batch_size, 
                                                                   class_loss, 
                                                                   plot_hist, 
                                                                   error_func)

        for subject in np.unique(test_subjects):
            for limb in np.unique(test_limbs):
                mask = (test_subjects == subject) & (test_limbs == limb)
                if np.sum(mask) > 0:
                    sens, spec, yi = youden_index(y_true[mask], y_pred[mask])
                    print(f"Limb: {limb}",
                          f"Subject: {subject}",
                          f"Youden Index: {yi}",
                          f"Sensitivity: {sens}",
                          f"Specificity: {spec}")
                    all_results.append({
                        'subject': subject,
                        'limb': limb,
                        'youden_index': yi,
                        'sensitivity': sens,
                        'specificity': spec,
                        'hyperparameters': best_hp
                    })

    return pd.DataFrame(all_results)

#function for nested cross validation for approach 2
def run_nested_cv_2(window_size, X, y, subjects, limbs, architecture, hp_grid, epochs, batch_size, 
                class_loss, plot_hist, error_func=None):
    all_results = []
    unique_subjects = np.unique(subjects)
    
    for test_subject in unique_subjects:
        tf.keras.backend.clear_session()
        remaining_subjects = unique_subjects[unique_subjects != test_subject]
        test_mask = subjects == test_subject
        X_test, y_test = X[test_mask], y[test_mask]

        for val_subject in remaining_subjects:
            train_subjects = remaining_subjects[remaining_subjects != val_subject]
            train_mask = subjects.isin(train_subjects)
            val_mask = subjects == val_subject
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            
            X_train_w, y_train_w, _, _ = create_sliding_windows(X_train, 
                                                                y_train, 
                                                                subjects[train_mask], 
                                                                limbs[train_mask], 
                                                                None, 
                                                                window_size)
            X_train_class_1 = X_train_w[y_train_w == 1]
            y_train_class_1 = y_train_w[y_train_w == 1]

            X_train_class_0 = X_train_w[y_train_w == 0]
            y_train_class_0 = y_train_w[y_train_w == 0]

            best_score = -np.inf
            best_hp = None
            
            for hp_set in ParameterGrid(hp_grid):
                model, encoder = build_autoencoder(architecture, hp_set,
                                                   input_shape=(X_train_class_1.shape[1], 
                                                                X_train_class_1.shape[2]),
                                                   num_classes=1, 
                                                   class_loss=class_loss)
                
                error_func = get_error_function(hp_set['loss'])
                
                history = train_model_without_class_loss(model, 
                                                         X_train_class_1, 
                                                         X_train_class_1, 
                                                         epochs, 
                                                         batch_size)
                thresh_errors = error_func(X_train_w, 
                                           model.predict(X_train_w, verbose=0))
                threshold = calculate_threshold(thresh_errors, y_train_w)
                
                X_val_w, y_val_w, val_subjects_w, val_limbs_w = create_sliding_windows(
                    X_val, y_val, subjects[val_mask], limbs[val_mask], None, window_size)
                
                val_errors = error_func(X_val_w, model.predict(X_val_w, verbose=0))
                val_pred = (val_errors <= threshold).astype(int)
                _, _, val_score = youden_index(y_val_w, val_pred)

                if val_score > best_score:
                    best_score = val_score
                    best_hp = hp_set
                    print(f"New best HP found: with score: {best_score:.3f}")
            
            all_train_mask = subjects.isin(remaining_subjects)
            X_all_train, y_all_train = X[all_train_mask], y[all_train_mask]
            
            X_all_train_w, y_all_train_w, _, _ = create_sliding_windows(X_all_train, 
                                                                        y_all_train, 
                                                                        subjects[all_train_mask], 
                                                                        limbs[all_train_mask], 
                                                                        None, 
                                                                        window_size)

            X_all_train_w1 = X_all_train_w[y_all_train_w == 1]
            y_all_train_w1 = y_all_train_w[y_all_train_w == 1]

 
            model, encoder = build_autoencoder(architecture, 
                                               best_hp, 
                                               input_shape=(X_all_train_w1.shape[1], 
                                                            X_all_train_w1.shape[2]), 
                                               num_classes=1, 
                                               class_loss=class_loss)
            
            history = train_model_without_class_loss(model, 
                                                     X_all_train_w1, 
                                                     X_all_train_w1, 
                                                     epochs, 
                                                     batch_size)
            
            all_train_errors = error_func(X_all_train_w, 
                                          model.predict(X_all_train_w, verbose=0))
            threshold = calculate_threshold(all_train_errors, y_all_train_w)
            
            X_test_w, y_test_w, test_subjects_w, test_limbs_w = create_sliding_windows(X_test, 
                                                                                       y_test, 
                                                                                       subjects[test_mask], 
                                                                                       limbs[test_mask], 
                                                                                       None, 
                                                                                       window_size)
            
            test_errors = error_func(X_test_w, model.predict(X_test_w, verbose=0))
            y_pred = (test_errors <= threshold).astype(int)
        
            for limb in np.unique(test_limbs_w):
                mask = test_limbs_w == limb
                if np.sum(mask) > 0:
                    sens, spec, yi = youden_index(y_test_w[mask], y_pred[mask])
                    print(f'test yi: {yi}')
                    best_hp_copy = best_hp.copy()
                    best_hp_copy['kernel_init'] = best_hp['kernel_init'].__class__.__name__
                    best_hp_copy['bias_init'] = best_hp['bias_init'].__class__.__name__
                    result = {
                                'subject': test_subject,
                                'limb': limb,
                                'youden_index': yi,
                                'sensitivity': sens,
                                'specificity': spec,
                                'threshold': threshold,
                                'val_subject': val_subject,
                                'hyperparameters': best_hp_copy
                             }
                    all_results.append(result)
    
    return pd.DataFrame(all_results)

#function for hyperparameter validation for approach 1 
def validate_hyperparameters(X, y, subjects, limbs, window_size, architecture, hp, epochs,
                           batch_size, class_loss, approach, error_func):
    logo_inner = LeaveOneGroupOut()
    scores = []
    error_func = get_error_function(hp['loss'])
    for train_idx, val_idx in logo_inner.split(X, groups=subjects):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_w, y_train_w, _, _ = create_sliding_windows(X_train, 
                                                            y_train, 
                                                            subjects.iloc[train_idx], 
                                                            limbs.iloc[train_idx], 
                                                            None, window_size
        )
        X_val_w, y_val_w, _, _ = create_sliding_windows(X_val, 
                                                        y_val, 
                                                        subjects.iloc[val_idx], 
                                                        limbs.iloc[val_idx], 
                                                        None, window_size )

        model, encoder = build_autoencoder(architecture, hp,
                                           input_shape=(X_train_w.shape[1], 
                                                        X_train_w.shape[2]), 
                                           num_classes=1, 
                                           class_loss=class_loss)

        if class_loss:
            history = train_model_with_class_loss(model, 
                                                  X_train_w, 
                                                  y_train_w, 
                                                  X_val_w, 
                                                  y_val_w, 
                                                  epochs, 
                                                  batch_size)
        else:
            history = train_model_without_class_loss(model, 
                                                     X_train_w, 
                                                     X_val_w, 
                                                     epochs, 
                                                     batch_size
            )
        
        if approach == 1:
            y_pred = run_rf(encoder, X_train_w, X_val_w, y_train_w)
        elif approach == 2:
            if class_loss:
                recon_train = error_func(X_train_w, 
                                         model.predict(X_train_w, verbose=0)[0])
                recon_val = error_func(X_val_w, 
                                       model.predict(X_val_w, verbose=0)[0])
            else:
                recon_train = error_func(X_train_w, 
                                         model.predict(X_train_w, verbose=0))
                recon_val = error_func(X_val_w, 
                                       model.predict(X_val_w, verbose=0))
            threshold = calculate_threshold(recon_train, y_train_w)
            y_pred = (recon_val <= threshold).astype(int)
        _, _, yi = youden_index(y_val_w, y_pred)
        scores.append(yi)

    mean_score = np.mean(scores)
    #print(f"Mean validation score: {mean_score:.3f}")
    return mean_score

#function for hyperparameter validation for approach 3a
def validate_hyperparameters_3a(X, y, subjects, limbs, window_size, architecture, hp, epochs,
                              batch_size, class_loss, error_func):
    logo_inner = LeaveOneGroupOut()
    scores = []
    error_func = get_error_function(hp['loss'])
    for train_idx, val_idx in logo_inner.split(X, groups=subjects):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_train_w, y_train_w, _, _ = create_sliding_windows(X_train, 
                                                            y_train, 
                                                            subjects.iloc[train_idx], 
                                                            limbs.iloc[train_idx], 
                                                            None, window_size)
        X_val_w, y_val_w, _, _ = create_sliding_windows(X_val, 
                                                        y_val, 
                                                        subjects.iloc[val_idx], 
                                                        limbs.iloc[val_idx], 
                                                        None, window_size)

        X_train_class_0 = X_train_w[y_train_w == 0]
        X_train_class_1 = X_train_w[y_train_w == 1]

        model0, _ = build_autoencoder(architecture, hp, 
                                      input_shape=(X_train_class_0.shape[1], 
                                                   X_train_class_0.shape[2]),
                                      num_classes=1, 
                                      class_loss=class_loss)

        model1, _ = build_autoencoder(architecture, hp, 
                                      input_shape=(X_train_class_1.shape[1], 
                                                   X_train_class_1.shape[2]),
                                      num_classes=1, 
                                      class_loss=class_loss)
        
        if class_loss:
            history0 = train_model_with_class_loss(model0, 
                                                   X_train_class_0, 
                                                   np.zeros((len(X_train_class_0))), 
                                                   X_val_w, 
                                                   y_val_w, 
                                                   epochs, 
                                                   batch_size)
            history1 = train_model_with_class_loss(model1, 
                                                   X_train_class_1, 
                                                   np.ones((len(X_train_class_1))), 
                                                   X_val_w, 
                                                   y_val_w, 
                                                   epochs, 
                                                   batch_size)
            recon_error_0 = error_func(X_val_w, 
                                       model0.predict(X_val_w, verbose=0)[0])
            recon_error_1 = error_func(X_val_w, 
                                       model1.predict(X_val_w, verbose=0)[0])
        else:
            history0 = train_model_without_class_loss(model0, 
                                                      X_train_class_0, 
                                                      X_val_w, 
                                                      epochs, 
                                                      batch_size)
            history1 = train_model_without_class_loss(model1, 
                                                      X_train_class_1, 
                                                      X_val_w, 
                                                      epochs, 
                                                      batch_size)
            recon_error_0 = error_func(X_val_w, 
                                       model0.predict(X_val_w, verbose=0))
            recon_error_1 = error_func(X_val_w, 
                                       model1.predict(X_val_w, verbose=0))

        y_pred = (recon_error_1 < recon_error_0).astype(int)
        _, _, yi = youden_index(y_val_w, y_pred)
        scores.append(yi)

    mean_score = np.mean(scores)
    #print(f"Mean validation score: {mean_score:.3f}")
    return mean_score

#function for hyperparameter validation for approach 3b
def validate_hyperparameters_3b(X, y, subjects, limbs, usetypes, window_size, architecture, hp, epochs,
                              batch_size, class_loss, error_func):
    logo_inner = LeaveOneGroupOut()
    scores = []
    error_func = get_error_function(hp['loss'])
    for train_idx, val_idx in logo_inner.split(X, groups=subjects):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        usetypes_train = usetypes.iloc[train_idx]
        usetypes_val = usetypes.iloc[val_idx]

        X_train_w, y_train_w, _, _, use_ids_train = create_sliding_windows(X_train, 
                                                                           y_train, 
                                                                           subjects.iloc[train_idx], 
                                                                           limbs.iloc[train_idx], 
                                                                           usetypes_train, 
                                                                           window_size)
        X_val_w, y_val_w, _, _, use_ids_val = create_sliding_windows(X_val, 
                                                                     y_val, 
                                                                     subjects.iloc[val_idx], 
                                                                     limbs.iloc[val_idx], 
                                                                     usetypes_val, window_size
        )

        autoencoders = {}
        history_list = {}
        recon_errors = []
        
        if class_loss:
            for use_type_val in [1, 2, 3]:
                X_train_use = X_train_w[use_ids_train == use_type_val]
                y_train_use = y_train_w[use_ids_train == use_type_val]
                autoencoder, _ = build_autoencoder(architecture, hp, 
                                                   input_shape=(X_train_use.shape[1], 
                                                                X_train_use.shape[2]), 
                                                   num_classes=1, 
                                                   class_loss=class_loss)
                history = train_model_with_class_loss(autoencoder, 
                                                      X_train_use, 
                                                      y_train_use, 
                                                      X_val_w, 
                                                      y_val_w, 
                                                      epochs, 
                                                      batch_size)
                history_list[use_type_val] = history
                autoencoders[use_type_val] = autoencoder
            
            for _, autoencoder in autoencoders.items():
                recon_error = error_func(X_val_w, 
                                         autoencoder.predict(X_val_w, verbose=0)[0])
                recon_errors.append(recon_error)
        else:
            for use_type_val in [1, 2, 3]:
                X_train_use = X_train_w[use_ids_train == use_type_val]
                autoencoder, _ = build_autoencoder(architecture, hp, 
                                                   input_shape=(X_train_use.shape[1], 
                                                                X_train_use.shape[2]), 
                                                   num_classes=1, 
                                                   class_loss=class_loss)
                history = train_model_without_class_loss(autoencoder, 
                                                         X_train_use, 
                                                         X_val_w, 
                                                         epochs, 
                                                         batch_size)
                history_list[use_type_val] = history
                autoencoders[use_type_val] = autoencoder
            
            for _, autoencoder in autoencoders.items():
                recon_error = error_func(X_val_w, 
                                         autoencoder.predict(X_val_w, verbose=0))
                recon_errors.append(recon_error)

        recon_errors = np.array(recon_errors)
        min_error_idx = np.argmin(recon_errors, axis=0)
        y_pred = (min_error_idx != 2).astype(int)

        _, _, yi = youden_index(y_val_w, y_pred)
        scores.append(yi)

    mean_score = np.mean(scores)
    #print(f"Mean validation score: {mean_score:.3f}")
    return mean_score

#for creating the sliding windows
def create_sliding_windows(data, targets, subject_nums, limb_type, use_type, window_size):
    features, labels, subjects, limbs, use_types = [], [], [], [], []
    
    if use_type is not None:
        for i in range(0, len(data) - window_size + 1, window_size):
            window = data[i:i + window_size]
            window_labels = targets[i:i + window_size]
            window_subjects = subject_nums[i:i + window_size]
            window_limbs = limb_type[i:i + window_size]
            window_usetype = use_type[i:i + window_size]
            features.append(window)
            labels.append(mode(window_labels, keepdims=True).mode[0])
            subjects.append(mode(window_subjects, keepdims=True).mode[0])
            limbs.append(mode(window_limbs, keepdims=True).mode[0])
            use_types.append(mode(window_usetype, keepdims=True).mode[0])
        return np.array(features), np.array(labels), np.array(subjects), np.array(limbs), np.array(use_types)
    else:
        for i in range(0, len(data) - window_size + 1, window_size):
            window = data[i:i + window_size]
            window_labels = targets[i:i + window_size]
            window_subjects = subject_nums[i:i + window_size]
            window_limbs = limb_type[i:i + window_size]
            
            features.append(window)
            labels.append(mode(window_labels, keepdims=True).mode[0])
            subjects.append(mode(window_subjects, keepdims=True).mode[0])
            limbs.append(mode(window_limbs, keepdims=True).mode[0])
        return np.array(features), np.array(labels), np.array(subjects), np.array(limbs)

#for finding the optimized model (no nested CV) - approach 1
def run_approach(window_size, X, y, subjects, limbs, architecture, hp, epochs, batch_size, class_loss, plot_hist, approach, error_func=None):
    logo = LeaveOneGroupOut()
    all_results = []
    
    for train_idx, test_idx in logo.split(X, groups=subjects):
        tf.keras.backend.clear_session()
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        y_true, y_pred, test_subjects, test_limbs = process_fold(
            X_train, X_test, y_train, y_test, subjects, limbs, train_idx, test_idx,
            window_size, architecture, hp, epochs, batch_size, class_loss, plot_hist, approach, error_func)
        
        for subject in np.unique(test_subjects):
            for limb in np.unique(test_limbs):
                mask = (test_subjects == subject) & (test_limbs == limb)
                if np.sum(mask) > 0:  
                    sens, spec, yi = youden_index(y_true[mask], y_pred[mask])
                    print(f"Limb: {limb}",
                          f"Subject: {subject}",
                          f"Youden Index: {yi}",
                          f"Sensitivity: {sens}",
                          f"Specificity: {spec}"
                          )
                    all_results.append({
                        'subject': subject,
                        'limb': limb,
                        'youden_index': yi,
                        'sensitivity': sens,
                        'specificity': spec
                    })
    
    return pd.DataFrame(all_results)


