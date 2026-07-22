import tensorflow as tf
from tensorflow.keras import layers, models, Model
import tensorflow.keras.backend as K

# 1. Base Feature Extraction Network
def build_base_network(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation=None)(x)  # 128-d embedding vector
    
    # L2 normalize embeddings
    embeddings = layers.Lambda(lambda t: K.l2_normalize(t, axis=1))(x)
    
    return Model(inputs, embeddings, name="Embedding_Network")

# 2. Euclidean Distance Layer
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# 3. Contrastive Loss Function
def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        # y_true: 1 for genuine pair, 0 for forged/dissimilar pair
        y_true = tf.cast(y_true, tf.float32)
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss

# 4. Build Complete Siamese Network
def build_siamese_network(input_shape=(128, 128, 1)):
    base_network = build_base_network(input_shape)
    
    input_a = layers.Input(shape=input_shape, name="reference_img")
    input_b = layers.Input(shape=input_shape, name="test_img")
    
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    distance = layers.Lambda(euclidean_distance, name="distance_layer")([processed_a, processed_b])
    
    return Model(inputs=[input_a, input_b], outputs=distance, name="Siamese_Network")