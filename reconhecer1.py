import cv2
import pickle
import time
from picamera2 import Picamera2

# Inicializa a câmera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(1)

# Carrega o modelo treinado
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("modelo_lbph.xml")

# Carrega os nomes das pessoas
with open("labels.pickle", "rb") as f:
    label_map = pickle.load(f)

# Carrega o classificador de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("Iniciando reconhecimento facial... pressione CTRL+C para parar.")

try:
    while True:
        # Captura frame da câmera
        frame = picam2.capture_array()

        # Converte para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detecta rostos
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))

            id_, conf = recognizer.predict(roi)

            if conf < 70:
                print(f"Rosto reconhecido: {label_map[id_]} (confiança: {conf:.2f})")

            else:
                print("Rosto desconhecido")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("Encerrando programa...")