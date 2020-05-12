import tensorflow as tf


def create_dataset(batch_size):
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    _ds_train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    _ds_test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    auto_num = tf.data.experimental.AUTOTUNE
    _ds_train = _ds_train.shuffle(len(train_x)).map(lambda x, y: (tf.cast(x, tf.float32) / 255., y), auto_num).batch(
        batch_size).prefetch(auto_num)

    _ds_test = _ds_test.map(lambda x, y: (tf.cast(x, tf.float32) / 255., y), auto_num).batch(
        batch_size).prefetch(auto_num)

    return _ds_train, _ds_test


class MnistModel(tf.keras.Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.f = tf.keras.layers.Flatten()
        self.d = tf.keras.layers.Dense(64, tf.keras.activations.relu)
        self.l = tf.keras.layers.Dense(10, tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        outputs = self.f(inputs)
        outputs = self.d(outputs)
        return self.l(outputs)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x)
            loss = self.compiled_loss(y, y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}  # 为了显示用的 1s 788us/step - loss: 0.3074 - acc: 0.9122


if __name__ == '__main__':
    ds_train, ds_test = create_dataset(32)
    model = MnistModel()
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["acc"])
    model.fit(ds_train, epochs=1)
