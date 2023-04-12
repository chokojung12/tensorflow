# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# load data
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# preprocessing data
train_images = train_images / 255.0
test_images = test_images / 255.0

# model load 
new_model = tf.keras.models.load_model('/app/input/dataset/fashion-mnist/mymodel')

#retrain
new_model.fit(train_images, train_labels, epochs=15)

# evaluation
test_loss, test_acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('\nNew Model Test accuracy:', test_acc)

# model save
new_model.save('/app/outputs/newmodel')
