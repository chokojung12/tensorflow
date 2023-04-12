# TensorFlow and tf.keras
import tensorflow as tf
import matplotlib.pyplot as plt

# Helper libraries
import numpy as np

# load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# preprocessing data
train_images = train_images / 255.0
test_images = test_images / 255.0

# create neural network(model)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# neural network(model) compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# train 100
model.fit(train_images, train_labels, epochs=10)


# evaluation
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# model save
model.save('/app/outputs/mymodel')
print('"Saved model to {}'.format('/app/outputs/mymodel'))
