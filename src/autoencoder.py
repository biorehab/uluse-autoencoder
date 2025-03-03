import tensorflow as tf
import numpy as np

# define the encoder layers for DCAE
def create_encoder_layers(x, num_filters, filter_sizes, pool_sizes, kernel_init, bias_init, activation, batch_norm = False):
    for i in range(len(num_filters)):
        if batch_norm:
            x = tf.keras.layers.Conv1D(filters=num_filters[i], 
                                        kernel_size=filter_sizes[i],
                                        kernel_initializer=kernel_init,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.005),
                                        bias_initializer=bias_init, 
                                        padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation=activation)(x)
        else: 
            x = tf.keras.layers.Conv1D(filters=num_filters[i], 
                                       kernel_size=filter_sizes[i],
                                       kernel_initializer=kernel_init,
                                       kernel_regularizer=tf.keras.regularizers.l2(0.005),
                                       bias_initializer=bias_init, 
                                       activation = activation,
                                       padding='same')(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=pool_sizes[i], padding='same')(x)
    return x

# define the decoder layers for DCAE
def create_decoder_layers(x, num_filters, filter_sizes, pool_sizes, kernel_init, bias_init, activation, batch_norm = False):
    for i in reversed(range(len(num_filters))):
        if batch_norm:
            x = tf.keras.layers.Conv1D(filters=num_filters[i], 
                                       kernel_size=filter_sizes[i],
                                       kernel_initializer=kernel_init,
                                       kernel_regularizer=tf.keras.regularizers.l2(0.005),
                                       bias_initializer=bias_init, 
                                       padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation=activation)(x)
        else: 
            x = tf.keras.layers.Conv1D(filters=num_filters[i], 
                                       kernel_size=filter_sizes[i],
                                       kernel_initializer=kernel_init,
                                       kernel_regularizer=tf.keras.regularizers.l2(0.005),
                                       bias_initializer=bias_init, 
                                       activation = activation,
                                       padding='same')(x)
        x = tf.keras.layers.UpSampling1D(size=pool_sizes[i])(x)
    return x

#latent space
def create_latent_space(x, input_shape, architecture):
    intermediate_shape = x.shape[1:]
    x = tf.keras.layers.Flatten()(x)
    latent = tf.keras.layers.Dense(architecture.get('latent_size', input_shape[0]),
                                    kernel_regularizer=tf.keras.regularizers.l2(0.005),
                                    name='latent_features')(x)
    return latent, intermediate_shape

#compile the model
def compile_model(encoder_input, decoded, latent, hp, class_loss):
    optimizer_dict = {
                      'nadam': tf.keras.optimizers.Nadam(hp['learning_rate']),
                      'adam': tf.keras.optimizers.Adam(hp['learning_rate']),
                      'sgd': tf.keras.optimizers.SGD(hp['learning_rate'])
                      }
    
    if class_loss:
        latent = tf.keras.layers.Dropout(0.2)(latent)
        classifier_output = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(latent)
        autoencoder = tf.keras.models.Model(inputs=encoder_input, outputs=[decoded, classifier_output])
        autoencoder.compile(optimizer=optimizer_dict.get(hp['optimizer']),
                            loss={'reconstructed': hp['loss'], 'classifier': 'binary_crossentropy'},
                            loss_weights={'reconstructed': 1, 'classifier': 0.1},
                            metrics={'classifier': 'accuracy'})
    else:
        autoencoder = tf.keras.models.Model(inputs=encoder_input, outputs=decoded)
        autoencoder.compile(optimizer=optimizer_dict.get(hp['optimizer']),
                            loss=hp['loss'])
    encoder = tf.keras.models.Model(inputs=encoder_input, outputs=latent)
    return autoencoder, encoder

#build the autoencoder
def build_autoencoder(architecture, hp, input_shape, num_classes, class_loss):
    tf.keras.utils.set_random_seed(42)
    
    encoder_input = tf.keras.layers.Input(shape=input_shape)
    x = create_encoder_layers(encoder_input, 
                              architecture['num_filters'], 
                              architecture['filter_sizes'], 
                              architecture['pool_sizes'], 
                              hp['kernel_init'], 
                              hp['bias_init'], 
                              hp['activation'], 
                              hp['batch_norm'])
    
    latent, intermediate_shape = create_latent_space(x, 
                                                     input_shape, 
                                                     architecture)
    
    x = tf.keras.layers.Dense(np.prod(intermediate_shape))(latent)
    x = tf.keras.layers.Reshape(intermediate_shape)(x)
    
    x = create_decoder_layers(x, 
                              architecture['num_filters'], 
                              architecture['filter_sizes'], 
                              architecture['pool_sizes'], 
                              hp['kernel_init'], 
                              hp['bias_init'], 
                              hp['activation'], 
                              hp['batch_norm'])
    
    decoded = tf.keras.layers.Conv1D(filters=input_shape[1], 
                                     kernel_size=10, 
                                     activation='linear',
                                     padding='same', 
                                     name='reconstructed')(x)
    
    return compile_model(encoder_input, 
                         decoded, 
                         latent, 
                         hp, 
                         class_loss)

