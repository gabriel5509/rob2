import cv2
import pickle
import time
import mediapipe as mp
from picamera2 import Picamera2
import numpy as np

# ========== CONFIGURAÇÃO DO RECONHECIMENTO FACIAL ==========
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("modelo_lbph.xml")

with open("labels.pickle", "rb") as f:
    label_map = pickle.load(f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ========== CONFIGURAÇÃO DO MEDIAPIPE ==========
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)

# ========== CONFIGURAÇÃO DA CÂMERA ==========
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(1)

# Estado atual do sistema
modo_imitacao = False
print("Sistema iniciado. Pressione CTRL+C para encerrar.\n")

try:
    while True:
        frame = picam2.capture_array()

        # Corrige cores se necessário
        if frame.shape[-1] == 4:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[-1] == 1:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

        if not modo_imitacao:
            # ====== MODO RECONHECIMENTO FACIAL ======
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200))
                id_, conf = recognizer.predict(roi)

                if conf < 70:
                    nome = label_map[id_]
                    print(f"Rosto reconhecido: {nome} (confiança: {conf:.2f})")
                    modo_imitacao = True
                    print(">>> Alternando para modo de imitação de movimentos.\n")
                    break
                else:
                    print("Rosto desconhecido")

            if len(faces) == 0:
                print("Nenhum rosto detectado.")

        else:
            # ====== MODO IMITAÇÃO DE MOVIMENTOS ======
            results_face = face_mesh.process(rgb_frame)
            results_hands = hands.process(rgb_frame)

            # Se detectar uma mão aberta → volta para modo de reconhecimento
            if results_hands.multi_hand_landmarks:
                for hand in results_hands.multi_hand_landmarks:
                    # Dedos abertos: conta os dedos estendidos (simplificado)
                    finger_tips = [8, 12, 16, 20]  # Pontas dos dedos
                    open_fingers = 0
                    for tip_id in finger_tips:
                        if hand.landmark[tip_id].y < hand.landmark[tip_id - 2].y:
                            open_fingers += 1

                    if open_fingers >= 4:
                        print("Mão aberta detectada → voltando ao modo de reconhecimento facial.\n")
                        modo_imitacao = False
                        break

            # Se não detectar mão, continua analisando os movimentos da cabeça/rosto
            if results_face.multi_face_landmarks and modo_imitacao:
                h, w, _ = rgb_frame.shape
                face = results_face.multi_face_landmarks[0]

                nose = face.landmark[1]
                left_eye = face.landmark[33]
                right_eye = face.landmark[263]
                chin = face.landmark[152]
                forehead = face.landmark[10]
                mouth_top = face.landmark[13]
                mouth_bottom = face.landmark[14]

                # Converter para coordenadas de pixel
                nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                left_eye_x = int(left_eye.x * w)
                right_eye_x = int(right_eye.x * w)
                chin_y = int(chin.y * h)
                forehead_y = int(forehead.y * h)
                mouth_open = abs(int((mouth_bottom.y - mouth_top.y) * h))

                # ----- Cabeça inclinada para cima/baixo -----
                if chin_y - forehead_y > h * 0.45:
                    print("Cabeça movida para baixo")
                elif chin_y - forehead_y < h * 0.35:
                    print("Cabeça movida para cima")

                # ----- Cabeça virada esquerda/direita -----
                face_center_x = (left_eye_x + right_eye_x) // 2
                if nose_x < face_center_x - 20:
                    print("Cabeça virada para esquerda")
                elif nose_x > face_center_x + 20:
                    print("Cabeça virada para direita")

                # ----- Boca aberta/fechada -----
                if mouth_open > 20:
                    print("Boca aberta")
                else:
                    print("Boca fechada")

            elif modo_imitacao:
                print("Nenhum rosto detectado, mas mantendo modo de imitação ativa...")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nEncerrando programa...")

finally:
    picam2.stop()
