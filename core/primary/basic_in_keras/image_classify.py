import tensorflow as tf

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

image_shape = (28, 28)
image_shape_conv = (28, 28, 1)


def create_data():
    (_train_x, _train_y), (_test_x, _test_y) = tf.keras.datasets.fashion_mnist.load_data()
    # [0,1]
    _train_x = _train_x / 255.
    _train_x = _train_x[..., tf.newaxis]
    _test_x = _test_x / 255.
    _test_x = _test_x[..., tf.newaxis]
    return (_train_x, _train_y), (_test_x, _test_y)


def create_model():
    _model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, 2, padding="same", activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(), input_shape=image_shape_conv),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(8, 3, 2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(),
                               activation="relu"),

        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(4, 3, 2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(),
                               activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])
    _model.compile("adam", tf.keras.losses.sparse_categorical_crossentropy, ["acc"])
    _model.summary()
    return _model


(train_x, train_y), (test_x, test_y) = create_data()
model = create_model()
model.fit(train_x, train_y, batch_size=32, epochs=10, validation_freq=0.2)
loss, acc = model.evaluate(test_x, test_y)
print("\nloss:{},acc:{}".format(loss, acc))

pred_y = model(test_x)
pred_y = pred_y[:20]
pred_y = tf.argmax(pred_y, axis=-1)
print(pred_y)
print(test_y[:20])
