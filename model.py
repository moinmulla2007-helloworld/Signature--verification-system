import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, BatchNormalization, Dropout
import tensorflow.keras.backend as K

def build_base_network(input_shape):
    """Lighter base CNN designed for small custom datasets."""
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.6)(x) 
    
    # Compress features
    embeddings = Lambda(lambda t: K.l2_normalize(t, axis=1))(x)
    return Model(inputs, embeddings, name="base_network")

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def build_siamese_network(input_shape=(128, 128, 1)):
    base_network = build_base_network(input_shape)
    
    input_a = Input(shape=input_shape, name="input_a")
    input_b = Input(shape=input_shape, name="input_b")
    
    feat_vec_a = base_network(input_a)
    feat_vec_b = base_network(input_b)
    
    distance = Lambda(euclidean_distance, name="distance_layer")([feat_vec_a, feat_vec_b])
    return Model(inputs=[input_a, input_b], outputs=distance, name="siamese_network")

def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    loss_match = (1 - y_true) * tf.square(y_pred)
    loss_mismatch = y_true * tf.square(tf.maximum(margin - y_pred, 0.0))
    return tf.reduce_mean(loss_match + loss_mismatch)