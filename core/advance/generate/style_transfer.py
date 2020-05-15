import tensorflow as tf
import os
import matplotlib.pyplot as plt


class StyleTransferModel(tf.keras.models.Model):
    def __init__(self,
                 pre_trained_model: tf.keras.Model,
                 content_feature_layers,
                 style_feature_layers):
        super().__init__(name="StyleTransferModel")
        self.pre_trained_model = pre_trained_model
        self.extract_content_layers = content_feature_layers
        self.extract_style_layers = style_feature_layers
        self.content_model = tf.keras.models.Model(
            self.pre_trained_model.inputs,
            [self.pre_trained_model.get_layer(layer).output for layer in content_feature_layers]
        )
        self.style_model = tf.keras.models.Model(
            self.pre_trained_model.inputs,
            [self.pre_trained_model.get_layer(layer).output for layer in style_feature_layers]
        )

    def gram_matrix(self, style_input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', style_input_tensor, style_input_tensor)
        input_shape = tf.shape(style_input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def call(self, inputs, training=None, mask=None):
        return {"content": self.content_model(inputs),
                "style": [self.gram_matrix(out) for out in self.style_model(inputs)]}


class Trainer(object):
    def __init__(self, model: tf.keras.models.Model):
        self.model = model
        self.opt_obj = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.metrics_loss = tf.keras.metrics.Mean()

    def loss_fn(self, outputs_pred, outputs_target):
        style_weight = 1e-2
        content_weight = 1e4
        # content　loss
        content_real = outputs_target["content"]
        content_pred = outputs_pred["content"]
        content_loss = tf.add_n([tf.reduce_mean((content_real - content_pred) ** 2)])
        content_loss *= content_weight / len(content_real)

        style_real = outputs_target["style"]
        style_pred = outputs_pred["style"]

        style_loss = tf.add_n(
            [tf.reduce_mean((real - pred) ** 2) for real, pred in zip(style_real, style_pred)])
        style_loss *= style_weight / len(style_real)
        return content_loss + style_loss

    def total_variation_loss(self, image):
        x_deltas, y_deltas = self.high_pass_x_y(image)
        return tf.reduce_mean(x_deltas ** 2) + tf.reduce_mean(y_deltas ** 2)

    def clip_0_1(self, image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    @tf.function
    def train_step(self, image, outputs_target):
        with tf.GradientTape() as tape:
            outputs_pred = self.model(image)
            loss = self.loss_fn(outputs_pred, outputs_target)
            loss += self.total_variation_loss(image)
        self.metrics_loss(loss)
        gradient = tape.gradient(loss, image)
        self.opt_obj.apply_gradients([(gradient, image)])
        image.assign(self.clip_0_1(image))

    def train(self, epochs, content_img_tensor, style_img_tensor):
        image = tf.Variable(content_img_tensor)
        content_target = self.model(content_img_tensor)["content"]
        style_target = self.model(style_img_tensor)["style"]
        outputs_target = {"content": content_target, "style": style_target}

        file_writer = tf.summary.create_file_writer("logs")
        with file_writer.as_default():
            for epoch in range(epochs):
                self.metrics_loss.reset_states()
                self.train_step(image, outputs_target)
                loss = self.metrics_loss.result()
                tf.summary.scalar("loss", loss, epoch)
                print("epoch:{}, loss:{}".format(epoch, loss))
                if epoch % 5 == 0:
                    tf.summary.image("img", image, epoch)

        print("done!")

    def high_pass_x_y(self,image):
        x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
        y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
        return x_var, y_var


def create_image_tensor(path, size):
    x = tf.io.read_file(path)
    x = tf.image.decode_jpeg(x)
    x = x[tf.newaxis, ...]
    x = tf.cast(x, tf.float32) / 255.
    x = tf.image.resize(x, size=(size, size))
    return x


def pre_process_img(input_img):
    x = input_img * 255
    x = tf.keras.applications.vgg19.preprocess_input(x)
    return x


if __name__ == '__main__':
    _pre_trained_model = tf.keras.applications.VGG19()
    _pre_trained_model.trainable = False
    # 内容层将提取出我们的 feature maps （特征图）
    _content_feature_layers = ['block5_conv2']
    # 我们感兴趣的风格层
    _style_feature_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1']
    style_transfer_model = StyleTransferModel(pre_trained_model=_pre_trained_model,
                                              content_feature_layers=_content_feature_layers,
                                              style_feature_layers=_style_feature_layers
                                              )

    _data_root = "data"
    _content_img_path = os.path.join(_data_root, "lion.jpg")
    # _style_img_path = os.path.join(_data_root, "art.jpg")
    # _style_img_path = os.path.join(_data_root, "woman-with-hat-matisse.jpg")
    _style_img_path = os.path.join(_data_root, "the_scream.jpg")
    # _style_img_path = os.path.join(_data_root, "starry-night.jpg")
    _content_img_tensor = create_image_tensor(_content_img_path, 224)
    _style_img_tensor = create_image_tensor(_style_img_path, 224)
    # 预处理输入图片
    # _content_img_tensor = pre_process_img(_content_img_tensor)
    # _style_img_tensor = pre_process_img(_style_img_tensor)
    trainer = Trainer(style_transfer_model)
    trainer.train(500, _content_img_tensor, _style_img_tensor)
