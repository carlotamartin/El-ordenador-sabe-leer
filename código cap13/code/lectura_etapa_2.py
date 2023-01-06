import cv2
import numpy as np

class Pizarra():

    #Constructor
    def __init__(self, zonaEscrituraLargoMin, zonaEscrituraLargoMax, zonaEscrituraAnchoMin, zonaEscrituraAnchoMax):
        self.zonaEscrituraLargoMin = zonaEscrituraLargoMin
        self.zonaEscrituraLargoMax = zonaEscrituraLargoMax
        self.zonaEscrituraAnchoMin = zonaEscrituraAnchoMin
        self.zonaEscrituraAnchoMax = zonaEscrituraAnchoMax

    def inicializarWebcam(self):
        print('Inicialización de la webcam')
        webCam = cv2.VideoCapture(0)
        if webCam.isOpened():
            largoWebcam = webCam.get(3)
            anchoWebcam = webCam.get(4)
            print('Resolución:' + str(largoWebcam) + " X " + str(anchoWebcam))
        else:
            print('ERROR')

        return webCam

    def detectarZonaEscritura(self, webCam):
        while True:
            (lecturaOK, frame) = webCam.read()

            (grabbed, frame) = webCam.read()
            tsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contornos_canny = cv2.Canny(gris, 30, 200)

            contornos = cv2.findContours(contornos_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

            for contorno in contornos:
                perimetro = cv2.arcLength(contorno, True)
                approx = cv2.approxPolyDP(contorno, 0.012 * perimetro, True)
                x, y, w, h = cv2.boundingRect(approx)

                #Se encuadra la zona de escritura en función de los parámetros de largo y ancho de la pizarra
                if len(approx) == 4 and h>self.zonaEscrituraAnchoMin and w>self.zonaEscrituraLargoMin and h<self.zonaEscrituraAnchoMax and w<self.zonaEscrituraLargoMax:

                    #Encuadre de la zona de escritura
                    area = cv2.contornoArea(contorno)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3);

                    # Captura de la imagen a partir de la zona de escritura con un margen interior (padding) de 10
                    # píxeles para aislar solo la letra
                    letra = gris[y + 10:y + h - 10, x + 10:x + w - 10]

                    # Se detectan los contornos de la letra con la ayuda del algoritmo Canny
                    cannyLetra = cv2.Canny(letra, 30, 200)
                    contornosLetra = cv2.findContornos(cannyLetra.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

                    # Si hay una letra d dibujada
                    if len(contornosLetra) > 5:

                        # Creación de una tabla para el almacenamiento de la imagen de la letra
                        capturaAlphabetTMP = np.zeros((400, 400), dtype=np.uint8)

                        # Se detecta el contorno de la letra
                        contornoLetra = max(contornosLetra, key=cv2.contourArea)

                        # Se obtiene el rectángulo que contiene la letra
                        xc, yc, wc, hc = cv2.boundingRect(contornoLetra)

                        for contornoLetra in contornosLetra:
                            area = cv2.contourArea(contornoLetra)
                            if area > 100:
                                cv2.drawContours(capturaAlphabetTMP, contornoLetra, -1, (255, 255, 255), 3)

                                #Se captura la letra y se guardan los valores de los píxeles de la zona capturada en una tabla
                                capturaLetra = np.zeros((400, 400), dtype=np.uint8)
                                capturaLetra = capturaAlphabetTMP[yc:yc + hc, xc:xc + wc]

                                #Se pueden capturar sombras en la zona de escritura provocando errores de
                                #reconocimiento. Si se dectecta una sombra, una de las dimensiones de la tabla de captura es
                                #igual a cero porque no se ha detectado ningún contorno de letra
                                visualizaciónLetraCapturada = True

                                if capturaLetra.shape[0] == 0 or capturaLetra.shape[1] == 0:
                                    visualizaciónLetraCapturada = False


                                #Si no se detecta una sombra, se muestra la letra capturada
                                if visualizaciónLetraCapturada:
                                    cv2.destroyWindow("Contorno de la letra")
                                    cv2.imshow('Captura de la letra', capturaLetra)

                                    #Redimensionamiento de la imagen
                                    newimage = cv2.resize(capturaLetra, (28, 28))
                                    newimage = np.array(newimage)
                                    newimage = newimage.astype('float32') /255
                                    newimage = newimage.reshape(1, 28, 28, 1)

            # Visualización de la imagen capturada por la webcam
            cv2.imshow("IMAGEN", frame)
            cv2.imshow("HSV", tsv)
            cv2.imshow("GRIS", gris)
            cv2.imshow("CANNY", contornos_canny)

            # Condición de salida del bucle While
            # > Tecla Escape para salir
            key = cv2.waitKey(1)
            if key == 27:
                break

        # Cierre de la ventana de la webcam
        webCam.release()
        cv2.destroyAllWindows()




def main():
    pizarra = Pizarra(540, 590, 300, 340)
    webCam = pizarra.inicializarWebcam()
    pizarra.detectarZonaEscritura(webCam)

if __name__ == '__main__':
    main()
