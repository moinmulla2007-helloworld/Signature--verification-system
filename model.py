import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda,
    BatchNormalization, LeakyReLU
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def euclidean_distance(vects):
    """
    Calculates the Euclidean distance between two feature vectors.
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    # K.epsilon() is added to prevent math errors if the distance is perfectly 0
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def build_base_network(input_shape):
    """
    The Convolutional Neural Network (CNN) backbone that extracts features from a single signature.
    """
    inputs = Input(shape=input_shape)

    # Feature Extraction Blocks
    # Switched ReLU -> LeakyReLU(0.1) throughout the backbone. Your inputs
    # are sparse binary ink masks (mostly 0 background), which is exactly
    # the condition that starves plain ReLU units of gradient and lets
    # them "die" (stuck outputting 0 for every image). LeakyReLU keeps a
    # small gradient alive for negative pre-activations so units can
    # recover instead of collapsing.
    x = Conv2D(64, (3, 3), kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    # --- THE ACTUAL COLLAPSE FIX ---
    # This Dense layer used to be `activation='relu'`. ReLU forces every
    # embedding coordinate to be >= 0, which confines every signature's
    # embedding to a single 256-D orthant. Two nonnegative unit vectors
    # can never be more than sqrt(2) apart and, in practice, cluster much
    # closer than that - so the network was fighting its own geometry to
    # satisfy margin=1.0, and the easiest local minimum it found was to
    # collapse everything toward one point instead.
    #
    # Leave this layer LINEAR (no activation). L2-normalizing a linear
    # embedding uses the *full* unit hypersphere (max distance = 2.0,
    # comfortably above your margin=1.0), which gives the network a much
    # easier, non-degenerate way to separate genuine from forged pairs.
    x = Dense(256, kernel_initializer='glorot_normal')(x)

    # --- L2 NORMALIZATION ---
    # Forces all extracted features onto a unit sphere, stabilizing the
    # distance calculation and capping the maximum possible distance at 2.0.
    x = Lambda(lambda tensors: K.l2_normalize(tensors, axis=1), name="l2_norm")(x)

    return Model(inputs, x, name="base_network")


def build_siamese_network(input_shape=(256, 256, 1)):
    """
    The Siamese architecture that passes two images through the base network and calculates their difference.
    """
    input_a = Input(shape=input_shape, name='input_a')
    input_b = Input(shape=input_shape, name='input_b')

    # Initialize the shared base network
    base_network = build_base_network(input_shape)

    # Pass both signatures through the exact same network (sharing weights)
    feat_a = base_network(input_a)
    feat_b = base_network(input_b)

    # Calculate the distance between the extracted features
    distance = Lambda(euclidean_distance, name='distance_layer')([feat_a, feat_b])

    model = Model(inputs=[input_a, input_b], outputs=distance, name="siamese_network")
    return model


def contrastive_loss(y_true, y_pred):
    """
    Custom loss function.
    y_true = 1 for Match, y_true = 0 for Forgery

    NOTE: this formula was already correct for your label convention -
    match_loss pulls genuine pairs toward D=0, forgery_loss pushes
    forged pairs past the margin. The collapse bug was in the
    architecture (see build_base_network), not here.
    """
    margin = 1.0

    y_true = tf.cast(y_true, tf.float32)

    # Loss for Genuine Matches (pull them closer to 0)
    match_loss = y_true * K.square(y_pred)

    # Loss for Forgeries (push them further than the margin)
    forgery_loss = (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))

    return K.mean(match_loss + forgery_loss)