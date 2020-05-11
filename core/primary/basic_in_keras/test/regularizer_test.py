import tensorflow as tf

regularizer = tf.keras.regularizers.l2()
tensor = tf.ones(shape=(5, 5))
out = regularizer(tensor)
print(out)

layer = tf.keras.layers.Dense(
    5, input_dim=5,
    use_bias=False,
    kernel_initializer='ones',
    kernel_regularizer=tf.keras.regularizers.l1(0.01),
    activity_regularizer=tf.keras.regularizers.l2(0.01))
model = tf.keras.models.Sequential([
    layer
])
tensor = tf.ones(shape=(5, 5)) * 2.0
out = model(tensor)
print(out)
print(layer.losses)
print(model.losses)

print("优化前trainable_variables ：\n")
print(model.trainable_variables)

with tf.GradientTape() as tape:
    loss = tf.reduce_sum(model.losses)
gradients = tape.gradient(loss, model.trainable_variables)
tf.keras.optimizers.Adam().apply_gradients(zip(gradients, model.trainable_variables))
print("优化后trainable_variables ：\n")
print(model.trainable_variables)
# regularizer 骑士就是往模型添加loss，然后根据反向传播原理，使用梯度下降方法更新相应的trainable_variables
