import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_trans_model(input_shape=(256, 128)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3)),
        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.LayerNormalization(),
        layers.TimeDistributed(
            layers.Dense(88, activation='sigmoid',
                         kernel_regularizer=regularizers.l2(1e-4),
                         name='piano_output')
        )
    ])
    return model

def binary_cross_entropy(pos_weight = 20.0, lambda_count = 0.05, lambda_sparse = 1e-3):
    def _loss(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)

        bce = -(pos_weight * y_true * tf.math.log(y_pred)
                + (1. - y_true) * tf.math.log(1. - y_pred))
        bce = tf.reduce_mean(bce)

        bin_pred = tf.cast(y_pred > 0.5, tf.float32)

        true_cnt = tf.reduce_sum(y_true, axis=-1)
        pred_cnt = tf.reduce_sum(bin_pred, axis=-1)

        count_penalty = tf.square(pred_cnt - true_cnt)
        count_penalty = tf.reduce_mean(count_penalty)
        sparse_penalty = tf.reduce_mean(y_pred)

        return bce + lambda_count * count_penalty + lambda_sparse * sparse_penalty

    return _loss


def compile_model(model, learning_rate: float = 3e-4, pos_weight: float = 20.0, lambda_count: float = 0.05, lambda_sparse: float = 1e-3):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=1.0),
        loss=binary_cross_entropy(pos_weight, lambda_count, lambda_sparse),
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
        ]
    )
