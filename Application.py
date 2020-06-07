""" Aplicación de detección y clasificación de caras """

# Imports
import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(
    '/home/alejandro/Escritorio/Prototype FaceReconigtion/Cascades/data/haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier(
    '/home/alejandro/Escritorio/Prototype FaceReconigtion/Cascades/data/haarcascades/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")
labels = {"name": 1}


def stopProgram():
    videoCap.release()
    cv2.destroyAllWindows()


# Capture frame-by-frame
def captureFrame():
    return videoCap.read()


def grayTransform():
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def cascadeDetect():
    return face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)


def drawNameInRectangle(id_, x, y):
    # Poner el nombre en el marco
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = labels[id_]
    color = (255, 255, 255)
    stroke = 2
    cv2.putText(frame, name, (x, y), font, 1,
                color, stroke, cv2.LINE_AA)


def personalizeFaceRectangle(x, y, widht, height):
    # Marco alrededor del rostro
    """ Como hemos escogido haarcascade_frontalface_alt2.xml nos detectará cuando tengamos la cara de frente a la cámara"""
    color = (87, 255, 51)
    stroke = 1
    cv2.rectangle(frame, (x, y), (widht, height), color, stroke)


def drawEyesRectangle(roi_gray, roi_color):
    # Marco de los ojos
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex, ey, ew, eh) in eyes:
        widht = ex+ew
        height = ey+eh
        color = (0, 255, 204)     # BGR
        stroke = 1
        cv2.rectangle(roi_color, (ex, ey), (widht, height), color, stroke)


def detectRegionOFInterestAndDrawRectangle():

    for (x, y, w, h) in faces:
        # Esta la región de interés
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        widht = x+w
        height = y+h

       # Vamos a utilizar el reconocedor
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 50:
            drawNameInRectangle(id_, x, y)

        personalizeFaceRectangle(x, y, widht, height)
        drawEyesRectangle(roi_gray, roi_color)

        # Guardar la ultima imagen detectada porla webcam en un archivo png
        img_item = "Image.png"
        cv2.imwrite(img_item, roi_color)


def drawFrame():
    # Mostrar el resultado del frame
    cv2.imshow('frame', frame)


# Vamos ha invertir el valor y conseguir el nombre
with open("Pickles/face-labels.pickle", 'rb') as f:
    readLabels = pickle.load(f)
    labels = {v: k for k, v in readLabels.items()}

""" El método VideoCapture de la API OpenCV, si le pasas el parámetro 0, 
 te captura el vídeo de la cámara por defecto del ordenador """
videoCap = cv2.VideoCapture(0)


while(True):
    ret, frame = captureFrame()
    gray = grayTransform()
    faces = cascadeDetect()

    detectRegionOFInterestAndDrawRectangle()
    drawFrame()

    # El frame se cerrara al presionar la s
    if cv2.waitKey(20) & 0xFF == ord('s'):
        break

# Una vez sales del bucle al presionar la s(stop), se destruyen todos los frames
stopProgram()
