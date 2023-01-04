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




