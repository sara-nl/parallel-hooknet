import tensorflow as tf


class HooknetLoss(tf.keras.losses.Loss):

    def __init__(self, loss_weights):
        super(HooknetLoss, self).__init__()
        self.loss_weights = loss_weights
        self.context_loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.target_loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    def calculate_loss(self, label, prediction):

        # prediction = [tf.dtypes.cast(x, tf.float32) for x in prediction]
        # label = [tf.dtypes.cast(x, tf.float32) for x in label]

        target_loss = self.loss_weights[0] * self.target_loss_function(label[0], prediction[0])
        context_loss = self.loss_weights[1] * self.context_loss_function(label[1], prediction[1])
        return target_loss + context_loss

    def call(self, y_true, y_pred):
        return self.calculate_loss(y_true, y_pred)
