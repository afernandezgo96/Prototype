""" Vamos a ver todas las imágenes y convertirlas en datos """
import cv2
import os
import numpy as np
from PIL import Image
import pickle

# Buscamos la localización de las fotos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Face-Images")

# Escogemos el tipo de filtro
face_cascade = cv2.CascadeClassifier(
    'Cascades/data/haarcascades/haarcascade_frontalface_alt2.xml')

# Vamos a hacer el reconocedor
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Creamos las listas donde almacenar fotos y etiquetas
current_id = 0
label_ids = {}
y_labels = []
x_train = []


def transformToGrayPhoto(path):
    # Convertimos la imagen a escala de grises, la reescalamos y luego a números
    pil_image = Image.open(path).convert("L")
    size = (550, 550)
    final_image = pil_image.resize(size, Image.ANTIALIAS)
    return np.array(final_image, "uint8")


def roiDetection(id_, image_array):
    # Buscamos la región de interés
    faces = face_cascade.detectMultiScale(
        image_array, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        # Esta la región de interés
        roi = image_array[y:y+h, x:x+w]
        x_train.append(roi)
        y_labels.append(id_)


def saveIds():
    # Vamos a guardar las IDs para poder utilizarlas
    with open("Pickles/face-labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)


def trainRecognizer():
    # Vamos a entrenar al reconocedor
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("recognizers/face-trainner.yml")


# Etiquetamos la foto según el nombre de la carpeta
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            # Añadimos el ID de la foto
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
                id_ = label_ids[label]

            image_array = transformToGrayPhoto(path)

            roiDetection(id_, image_array)
saveIds()
trainRecognizer()
