# TensorFlow and tf.keras
import tensorflow as tf
# Helper libraries
import os
import PIL
import PIL.Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(tf.__version__)
cifar = tf.keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar.load_data()

train_labels = train_labels.reshape(len(train_labels),)
test_labels = test_labels.reshape(len(test_labels),)
#loading class labels
df = pd.read_csv("data/trainLabels.csv")
class_names = df['label'].unique()
class_names.sort()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.get_cmap("binary"))                             #https://stackoverflow.com/questions/51452112/how-to-fix-cm-spectral-module-matplotlib-cm-has-no-attribute-spectral
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
def plot(i, prediction, true_label, img):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, prediction[i], true_label, img)
    plt.subplot(1,2,2)
    plot_value_array(i, prediction[i],  true_label)
    plt.show()
def predict(i, prediction_array, true_label, img_array):
    img = img_array[i]
    #print(img.shape)
    img = (np.expand_dims(img,0))
    #print(img.shape)
    predictions_single = probability_model.predict(img)
    #print(predictions_single)
    print(class_names[np.argmax(predictions_single[0])])
    plot(i, prediction_array, true_label, img_array)
  
predict(0, predictions, test_labels, test_images)
predict(10, predictions, test_labels, test_images)
# predict(159, predictions, test_labels, test_images)
# predict(195, predictions, test_labels, test_images)
# predict(519, predictions, test_labels, test_images)
# predict(591, predictions, test_labels, test_images)
# predict(915, predictions, test_labels, test_images)
# predict(951, predictions, test_labels, test_images)
# predict(357, predictions, test_labels, test_images)
# predict(375, predictions, test_labels, test_images)
# predict(537, predictions, test_labels, test_images)
# predict(573, predictions, test_labels, test_images)
# predict(735, predictions, test_labels, test_images)
# predict(753, predictions, test_labels, test_images)


