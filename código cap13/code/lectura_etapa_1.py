import cv2 #importamos la libreria opencv


#-----------------------------------------------------------------------------------------

class Pizarra ():

    #Constructor de la clase
    def __init__(self, zonaEscrituraLargoMin, zonaEscrituraLargoMax, zonaEscrituraAnchoMin, zonaEscrituraAnchoMax):
        self.zonaEscrituraLargoMin = zonaEscrituraLargoMin
        self.zonaEscrituraLargoMax = zonaEscrituraLargoMax
        self.zonaEscrituraAnchoMin = zonaEscrituraAnchoMin
        self.zonaEscrituraAnchoMax = zonaEscrituraAnchoMax


    #getters y setters
    def getZonaEscrituraLargoMin(self):
        return self.zonaEscrituraLargoMin

    def setZonaEscrituraLargoMin(self, zonaEscrituraLargoMin):
        self.zonaEscrituraLargoMin = zonaEscrituraLargoMin

    def getZonaEscrituraLargoMax(self):
        return self.zonaEscrituraLargoMax

    def setZonaEscrituraLargoMax(self, zonaEscrituraLargoMax):
        self.zonaEscrituraLargoMax = zonaEscrituraLargoMax

    def getZonaEscrituraAnchoMin(self):
        return self.zonaEscrituraAnchoMin

    def setZonaEscrituraAnchoMin(self, zonaEscrituraAnchoMin):
        self.zonaEscrituraAnchoMin = zonaEscrituraAnchoMin

    def getZonaEscrituraAnchoMax(self):
        return self.zonaEscrituraAnchoMax

    def setZonaEscrituraAnchoMax(self, zonaEscrituraAnchoMax):
        self.zonaEscrituraAnchoMax = zonaEscrituraAnchoMax


    #Función para inicializar la webcam
    def inicializarWebcam(self):
        webCam = cv2.VideoCapture(0)
        if webCam.isOpened():
            largoWebcam = webCam.get(3)
            anchoWebcam = webCam.get(4)
            print('Resolución:' + str(largoWebcam) + " X " + str(anchoWebcam))
        else:
            print('ERROR')
        return webCam


    #Función para detectar la pizarra
    def detectarPizarra(self, webCam):

        print('Detectando pizarra')
        while True:
            # Captura de la imagen en la variable Frame
            # La variable lecturaOK es igual a True si la función read() está operativa
            (lecturaOK, frame) = webCam.read()
            (grabbed, frame) = webCam.read()
            tsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contornos_canny = cv2.Canny(gris, 30, 200)

            contornos = cv2.findContours(contornos_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

            for c in contornos:
                perimetro = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.012 * perimetro, True)
                x, y, w, h = cv2.boundingRect(approx)

                #Se encuadra la zona de escritura en función de los parámetros de largo y ancho de la pizarra
                if len(approx) == 4 and h>self.zonaEscrituraAnchoMin and w>self.zonaEscrituraLargoMin and h<self.zonaEscrituraAnchoMax and w<self.zonaEscrituraLargoMax:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.imshow('IMAGEN', frame)
            cv2.imshow('HSV', tsv)
            cv2.imshow('GRIS', gris)
            cv2.imshow('CANNY', contornos_canny)

            # Condición de salida del bucle While
            # > Tecla Escape para salir
            key = cv2.waitKey(1)
            if key == 27:
                break

        # Cierre de la ventana
        webCam.release()
        cv2.destroyAllWindows()

def main():
    pizarra = Pizarra(540, 590, 300, 340)
    print(pizarra.getZonaEscrituraLargoMin())
    print(pizarra.getZonaEscrituraLargoMax())
    print(pizarra.getZonaEscrituraAnchoMin())
    print(pizarra.getZonaEscrituraAnchoMax())

    print('Inicialización de la webcam')
    webCam = pizarra.inicializarWebcam()
    pizarra.detectarPizarra(webCam)



if __name__ == '__main__':
    main()