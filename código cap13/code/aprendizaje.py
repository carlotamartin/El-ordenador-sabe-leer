#Importamos las librerías necesarias

from __future__ import print_function
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

from mnist import MNIST #Librería para cargar las imágenes.
#Recordad que se debe instalarla con el comando pip install python-mnist

import numpy as np

import matplotlib.pyplot as plt

class Aprendizaje ():

    #Constructor de la clase
    def __init__(self, largoImagen, anchoImagen, cantidadImagenes, cantidadEtiquetas, cantidadCaracteres, epochs, batch_size):
        self.largoImagen = largoImagen
        self.anchoImagen = anchoImagen
        self.cantidadImagenes = cantidadImagenes
        self.cantidadEtiquetas = cantidadEtiquetas
        self.cantidadCaracteres = cantidadCaracteres
        self.epochs = epochs
        self.batch_size = batch_size


    #Carga de las imágenes
    def cargaImagenes(self):
        emnist_data = MNIST(path='datas\\', return_type='numpy')
        emnist_data.select_emnist('letters')
        Imagenes, Etiquetas = emnist_data.load_training()

        return Imagenes, Etiquetas


    #Conversión de las imágenes y etiquetas en tabla numpy
    def conversionImagenes(self, Imagenes, Etiquetas):

        Imagenes = np.asarray(Imagenes)
        Etiquetas = np.asarray(Etiquetas)

        return Imagenes, Etiquetas

    #Transformación de las tablas de imágenes, para que sean de 28*28
    def transformacionImagenes(self, Imagenes, Etiquetas):

        Imagenes = Imagenes.reshape(self.cantidadImagenes, self.anchoImagen, self.largoImagen)
        Etiquetas= Etiquetas.reshape(self.cantidadEtiquetas, 1)

        return Imagenes, Etiquetas

    #Visualización de la imagen N.° 70000
    def visualizacionImagen(self, Imagenes, Etiquetas):

        plt.imshow(Imagenes[70000], cmap='gray')
        plt.title('Etiqueta: '+str(Etiquetas[70000]))
        plt.show()

        Etiquetas = Etiquetas-1
        print("Etiqueta de la imagen N.° 70000...")
        print(Etiquetas[70000])


class validacion():

    def __init__(self, Imagenes, Etiquetas, epochs, batch_size):
        self.imagenes_aprendizaje, self.imagenes_validacion, self.etiquetas_aprendizaje, self.etiquetas_validacion = train_test_split(Imagenes, Etiquetas, test_size=0.25, random_state=42)
        self.epochs = epochs
        self.batch_size = batch_size

    def validacion(self, anchoimagen, largoimagen, cantidad_de_clases):
        self.imagenes_aprendizaje = self.imagenes_aprendizaje.reshape(self.imagenes_aprendizaje.shape[0], anchoimagen, largoimagen, 1)
        self.imagenes_validacion = self.imagenes_validacion.reshape(self.imagenes_validacion.shape[0], anchoimagen, largoimagen, 1)

        self.imagenes_aprendizaje = self.imagenes_aprendizaje.astype('float32')
        self.imagenes_validacion = self.imagenes_validacion.astype('float32')
        self.imagenes_aprendizaje /= 255
        self.imagenes_validacion /= 255

        self.etiquetas_aprendizaje = keras.utils.to_categorical(self.etiquetas_aprendizaje, cantidad_de_clases)
        self.etiquetas_validacion = keras.utils.to_categorical(self.etiquetas_validacion, cantidad_de_clases)

        return self.imagenes_aprendizaje, self.imagenes_validacion, self.etiquetas_aprendizaje, self.etiquetas_validacion

class redNeuronal():
    def __init__(self):
        self.redCNN = Sequential()

    def red(self, anchoimagen, largoimagen, cantidad_de_clases):
        self.redCNN.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(anchoimagen, largoimagen, 1)))
        self.redCNN.add(Conv2D(64, (3, 3), activation='relu'))
        self.redCNN.add(MaxPooling2D(pool_size=(2, 2)))
        self.redCNN.add(Dropout(0.25))
        self.redCNN.add(Flatten())
        self.redCNN.add(Dense(128, activation='relu'))
        self.redCNN.add(Dropout(0.5))
        self.redCNN.add(Dense(cantidad_de_clases, activation='softmax'))

        return self.redCNN

    def compilacion(self, redCNN):
        redCNN.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=keras.optimizers.Adadelta(),
                        metrics=['accuracy'])

        return redCNN

    def entrenamiento(self, redCNN, imagenes_aprendizaje, etiquetas_aprendizaje, imagenes_validacion, etiquetas_validacion, epochs, batch_size):
        redCNN.fit(imagenes_aprendizaje, etiquetas_aprendizaje,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(imagenes_validacion, etiquetas_validacion))

        return redCNN

    def guardado(self, redCNN):
        redCNN.save('modelo/modelo.h5')


    def evaluacion(self, redCNN, imagenes_validacion, etiquetas_validacion):
        score = redCNN.evaluate(imagenes_validacion, etiquetas_validacion, verbose=0)
        print('Pérdida:', score[0])
        print('Precisión:', score[1])


def main():
    aprendizaje = Aprendizaje(28, 28, 124800, 124800, 26, 12, 128)

    Imagenes, Etiquetas = aprendizaje.cargaImagenes()

    print("Cantidad de imágenes ="+str(len(Imagenes)))
    print("Cantidad de etiquetas ="+str(len(Etiquetas)))

    Imagenes, Etiquetas = aprendizaje.conversionImagenes(Imagenes, Etiquetas)

    print("Transformación de las tablas de imágenes...")
    Imagenes, Etiquetas = aprendizaje.transformacionImagenes(Imagenes, Etiquetas)

    print("Visualización de la imagen N.° 70000...")
    aprendizaje.visualizacionImagen(Imagenes, Etiquetas)




