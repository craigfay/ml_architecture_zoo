# Multinomial classification

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

mnist = tf.keras.datasets.mnist


# x_train is an array of inputs, y_train is an array of correct outputs
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Values in the input matrix are are between 0 and 255,
# so we'll divide to get values between 0 and 1 instead.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Tell keras which float type to use
tf.keras.backend.set_floatx('float64')

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)), #input images are 28x28 pixels
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])

# A vector of "logits", or probability that the input belongs
# to an output class. Applying a final 10 node dense layer
# indicates that we want to differentiate between 10 classes if input.
non_normalized_predictions = model(x_train[:1]).numpy()
predictions = tf.nn.softmax(non_normalized_predictions).numpy()

# This loss function needs a logit vector, and the index of the correct answer.
# With  loss functions, near-zero values indicate correctness.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# What does Model.compile() do?
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# "Model.fit method adjusts the model parameters to minimize the loss"
# Which parameters are being adjusted?
model.fit(x_train, y_train, epochs=5)

