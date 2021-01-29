# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 11:23:31 2021

@author: Jerzy Rzesniowiecki & Szymon Maj
"""
#Ustawienie importow
import os
import matplotlib.pyplot as plt
import tensorflow as tf

#Dane treningowe
train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

#Dane Testowe
test_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
test_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                  origin=test_url)


#Opisanie kolumn w datasecie
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

#Podzial na nazwy cech oraz nazwy wlasciwe irysow
feature_names = column_names[:-1]
label_name = column_names[-1]
class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']


#Ustalenie rozmiaru elementow w jednej probce trenigowej
batch_size = 32

#Wykorzystanie funkcji tf.data.experimental.make_csv_dataset do przystosowania danych traningowych oraz testowych
#Standardowo funkcja make_csv_dataset miesza dane więc dla modelu trenigowego nie musimy deklarowac shuffle=True
#Pierwszy input to plik/lista plikow z zawierajacych wpisy CSV
#batch_size Ilosc wpisow polaczonych w jeden batch
#column_names jesli nie jest zaden podany to pobierany jest z pierwszego wpisu w datasetcie
#label_name podanie tej zmiennej powoduje wykorzystanie przewidywalnego formatu poprzez wykorzystanie podanych danych jako osobnego tensora
#num_epochs ilosc epok do wykonania
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

test_dataset = tf.data.experimental.make_csv_dataset(
    test_fp,
    batch_size,
    column_names=column_names,
    label_name='species',
    num_epochs=1,
    shuffle=False)


#Weryfikacja danych datasetu trenigowego na grafie poprzez iteracje

# features, labels = next(iter(train_dataset))

# plt.scatter(features['petal_length'],
#             features['sepal_length'],
#             c=labels,
#             cmap='viridis')

# plt.xlabel("Petal length")
# plt.ylabel("Sepal length")
# plt.show()



def pack_features_vector(features, labels):
  #Zebranie wpisow w jedna tablice.
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)

test_dataset = test_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

#Grupowanie pojedynczych warstw w model keras
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(3)
])

predictions = model(features)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#Metoda do obliczenia strat
def loss(model, x, y, training):
  # training=training deklaracja treningu potrzebna jesli wykorzystany jest dropout
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels, training=False)
print("Loss test: {}".format(l))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels, training=True).numpy()))

train_loss_results = []
train_accuracy_results = []

num_epochs = 345 #Przy tej liczbie epok znajduje w wysokiej czestotliwosci najcelniejszy

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Petla treningowa z wykorzystaniem batchy 32
  for x, y in train_dataset:
    # Optymalizacja modelu
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    #Postep
    epoch_loss_avg.update_state(loss_value)  # Dodanie obecnej straty z batch
    # Porownanie wyliczonego labela do faktycznego podanego
    # training=training deklaracja treningu potrzebna jesli wykorzystany jest dropout
    epoch_accuracy.update_state(y, model(x, training=True))

  # Koniec epoki
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))
    
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()

test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
    logits = model(x, training=False)
    prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

tf.stack([y,prediction],axis=1)

#Dataset na twardo do predykcji w celach porównawczych

predict_dataset = tf.convert_to_tensor([
    [5.4,3.4,1.7,0.2],  #0
    [5.6,3.0,4.5,1.5],  #1
    [6.3,2.9,5.6,1.8],  #2
    [6.3,2.5,4.9,1.5],  #1
    [5.8,2.7,3.9,1.2],  #1
    [6.1,3.0,4.6,1.4],  #1
    [5.2,4.1,1.5,0.1],  #0
    [6.7,3.1,4.7,1.5],  #1
    [6.7,3.3,5.7,2.5],  #2
    [6.4,2.9,4.3,1.3],  #1
])

#Odpytanie modelu o przypisanie na podstawie danych parametrow do kategori irysow
predictions = model(predict_dataset, training=False)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))