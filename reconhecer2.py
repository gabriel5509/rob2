import cv2
import pickle
from picamera2 import Picamera2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Inicializar câmera e reproduz o vídeo na tela
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# Carregar classificador Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Carregar modelo treinado
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("modelo_lbph.xml")

# Carregar labels
with open("labels.pickle", "rb") as f:
    label_map = pickle.load(f)

# Configurar janela matplotlib
plt.ion()
fig, ax = plt.subplots()

try:
    while True:
        # Captura frame
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detecta rostos
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (200, 200))

            # Reconhece
            id_, conf = recognizer.predict(roi_resized)

            if conf < 60:  # valor menor = mais confiável
                nome = label_map.get(id_, "Desconhecido")
            else:
                nome = "Desconhecido"

            # Desenhar retângulo e nome
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, nome, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Mostrar no matplotlib
        ax.clear()
        ax.imshow(frame)
        ax.set_axis_off()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("Encerrando...")
finally:
    picam2.stop()