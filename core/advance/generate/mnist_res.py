import tensorflow as tf


def create_ds(batch_size):
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    auto_num = tf.data.experimental.AUTOTUNE
    _ds_train = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(len(train_x)).map(
        lambda x, y: (tf.cast(x[..., tf.newaxis], tf.float32) / 255., y), auto_num).batch(batch_size).prefetch(
        auto_num)
    _ds_test = tf.data.Dataset.from_tensor_slices((test_x, test_y)).map(
        lambda x, y: (tf.cast(x[..., tf.newaxis], tf.float32) / 255., y), auto_num).batch(batch_size).prefetch(auto_num)
    print(next(iter(_ds_train)))
    return _ds_train, _ds_test


def create_model():
    input_img = tf.keras.layers.Input(shape=(28, 28, 1))
    x = res_block(input_img)
    x = res_block(x, expand=8)
    x = res_block(x)
    x = res_block(x, expand=8)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)(x)
    _model = tf.keras.models.Model(input_img, x)
    _model.compile("adam", tf.keras.losses.sparse_categorical_crossentropy, ["acc"])
    return _model


def res_block(x, kernel_size=3, strides=1, expand=0):
    origin_x = x
    in_channel = x.shape[-1]
    if expand:
        strides = 2
        x = tf.keras.layers.Conv2D(filters=in_channel * expand / 2, kernel_size=kernel_size, strides=strides,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters=in_channel * expand, kernel_size=kernel_size,
                                   padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        origin_x = tf.keras.layers.Conv2D(filters=in_channel * expand, kernel_size=kernel_size, strides=strides,
                                          padding="same")(origin_x)
        origin_x = tf.keras.layers.BatchNormalization()(origin_x)

        return tf.keras.layers.add(inputs=[x, origin_x])
    else:
        x = tf.keras.layers.Conv2D(filters=in_channel * 4, kernel_size=kernel_size, strides=strides, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(filters=in_channel, kernel_size=kernel_size, strides=strides, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.add(inputs=[x, origin_x])
        return x


def create_simple_model():
    _model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, 3, 2, "same", activation="relu"),
        tf.keras.layers.Conv2D(4, 3, 2, "same", activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, tf.keras.activations.softmax)
    ])
    _model.compile("adam", tf.keras.losses.sparse_categorical_crossentropy, ["acc"])
    return _model


if __name__ == '__main__':
    ds_train, ds_test = create_ds(32)
    model = create_simple_model()
    # model = create_model()
    #model.summary()
    # tf.keras.utils.plot_model(model, "mnist_res.png", show_shapes=True)
    model.fit(ds_train, epochs=10, validation_data=ds_test)
