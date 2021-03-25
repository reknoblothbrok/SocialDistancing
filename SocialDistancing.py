#Importar las librerías necesarías 
import numpy as np
import time
import cv2
import math
import imutils

#Cargar lista de nombres del DataSet de YoloV3
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

#Cargo el modelo del directorio YOLO-COCO
#Modelo a 220FPS
weightsPath = "yolov3-tiny.weights"
configPath = "yolov3-tiny.cfg"
#Modelo a 45 FPS
#weightsPath = "yolov3-320.weights"
#configPath = "yolov3-320.cfg"
#Modelo a 20 FPS
#weightsPath = "yolov3-spp.weights"
#configPath = "yolov3-spp.cfg"

print("Cargando el modelo ...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Inicializo la camara
print("Inicializando camara ...")
cap = cv2.VideoCapture(0)
#cap =cv2.VideoCapture('prueba.mp4')

#Se inicia un ciclo en el que la Red neuronal tomara una predicción por fotograma
while(cap.isOpened()):
    #Leemos el siguiente fotograma del archivo
    ret, image = cap.read()
    #Reajusto el tamaño del marco y reajustamos sus dimensiones
    image = imutils.resize(image, width=800)
    (H, W) = image.shape[:2]
    #obtengo el nombre de las capas de salida del DataSet
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #Preproceso las imagenes para obtener una predicción
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    #Inicializamos una lectura del tiempo que se tarda cada predicción
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("Prediction time/frame : {:.6f} seconds".format(end - start))
    #Inicilizo la probabilidad mínima y cuadros delimitantes
    boxes = []
    #Filtra la probabilidad de predicción
    confidences = []
    #Clase asociada
    classIDs = []
    #Bucle en cada una de las salidas de las capas
    for output in layerOutputs:
        #Bucle en cada una de las detecciones
        for detection in output:
            #Extrae la identificación de clase y la probabilidad de detección
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            #Filtra las detecciones, asegurandonos de que solo detecte personas
            if confidence > 0.5 and classID == 0:
                #Escalamos las coordenadas del cuadro delimitador con el tamaño de la imagen
                #Tomando en cuenta que YOLO devuelve las coordenadas centrales X, Y del cuadro
                #delimitador y posteriormente el ancho y altura del cuadro
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                #Usamos las coordenadas (X, Y) del centro del cuadro para delimitar la parte superior
                #y la esquina izquierda
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                #Actualizamos las coordenadas de los cuadros delimitadores
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    #Aplicamos non-maxima suppression para eliminar las predicciones deviles            
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
    ind = []
    for i in range(0,len(classIDs)):
        if(classIDs[i]==0):
            ind.append(i)
    a = []
    b = []
    #Nos aseguramos de que almenos haya una predicción
    if len(idxs) > 0:
            for i in idxs.flatten():
                #Extraemos las coordenadas del cuadro delimitador
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                a.append(x)
                b.append(y)            
    distance=[] 
    nsd = []
    #Bucle sobre el triangulo superior de las distancias
    #Calculamos la distancia eucladiana con los pixeles de la imagen
    for i in range(0,len(a)-1):
        for k in range(1,len(a)):
            if(k==i):
                break
            else:
                x_dist = (a[k] - a[i])
                y_dist = (b[k] - b[i])
                d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                distance.append(d)
                #Comprobamos la distancia entre dos puntos
                if(d <=220):
                    nsd.append(i)
                    nsd.append(k)
                nsd = list(dict.fromkeys(nsd))
                print(nsd)
    #Inicializamos el recuadro rojo si la distancia es menor de 1.5m
    color = (0, 0, 255) 
    for i in nsd:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        #Dibujamos el recuadro al rededor de la persona detectada y le agregamos titulo al texto
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "Alerta"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    #Inicializamos el recuadro color verde si la distancia es mayor a 1.5m
    color = (0, 255, 0) 
    if len(idxs) > 0:
        for i in idxs.flatten():
            if (i in nsd):
                break
            else:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                #Dibujamos el recuadro al rededor de la persona detectada y le agregamos titulo al texto
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = 'Normal'
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)   
    #Mostramos la imagen de salida, ya sea en directo o no
    cv2.imshow("Detector de distanciamiento social", image)
    #Indicamos que con la letra "Q" cerramos la ventana
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#Destruimos todas las ventanas abiertas
cap.release()
cv2.destroyAllWindows()