"""
Created on Sun Jan 24 12:33:21 2021

@author: Jerzy Rzesniowiecki & Szymon Maj
"""
# TensorFlow and tf.keras
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
#Zaciaganie danych z bazy danych w sieci
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#Kodowanie kolo jest w skali 0 do 255, zmiana wartosci do skali 0 do 1
train_images = train_images / 255.0
test_images = test_images / 255.0

#Splaszczanie modelu i dodawanie warstw sieci
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Trening
model.fit(train_images, train_labels, epochs=10)

#Testowanie
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#Predykcja
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
    
def predict(i, prediction_array, true_label, img_array):
    '''
    Parameters
    ----------
    i : TYPE INTIGER
        
    prediction_array : TYPE ARRAY OF FLOAT
        
    true_label : TYPE ARRAY OF UINT8
        
    img_array : TYPE ARRAY OF FLOAT
        

    Returns
    -------
    None.

    '''
    def plot_image(i, predictions_array, true_label, img):
        '''
        Parameters
        ----------
        i : TYPE INTIGER
            
        predictions_array : TYPE ARRAY OF FLOAT
            
        true_label : TYPE ARRAY OF UINT8
            
        img : TYPE ARRAY OF FLOAT
            
    
        Returns
        -------
        None.
    
        '''
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
        '''
        Parameters
        ----------
        i : TYPE INTIGER
            
        predictions_array : TYPE ARRAY OF FLOAT 
            
        true_label : TYPE ARRAY OF UINT8
            
    
        Returns
        -------
        None.
    
        '''
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
        '''
        Parameters
        ----------
        i : TYPE INTIGER
            
        prediction : TYPE ARRAY OF FLOAT
            
        true_label : TYPE ARRAY OF UINT8
            
        img : TYPE ARRAY OF FLOAT
            
    
        Returns
        -------
        None.
    
        '''
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plot_image(i, prediction[i], true_label, img)
        plt.subplot(1,2,2)
        plot_value_array(i, prediction[i],  true_label)
        plt.show()
    
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
