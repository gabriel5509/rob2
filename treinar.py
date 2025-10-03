import cv2
import os
import numpy as np

# Caminho onde ficam as pastas das pessoas
data_path = "faces"
people = os.listdir(data_path)  # lista de subpastas (ex: gabriel, maria)

labels = []   # vai guardar os números (0,1,2...) de cada pessoa
faces = []    # vai guardar as imagens dos rostos
label_map = {}  # mapeia número -> nome

# Percorre cada pasta (cada pessoa)
for idx, person in enumerate(people):
    label_map[idx] = person  # ex: 0->gabriel, 1->maria
    person_path = os.path.join(data_path, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # lê em preto e branco
        if img is None:
            continue
        faces.append(img)      # adiciona a imagem
        labels.append(idx)     # adiciona o número correspondente à pessoa

# Converte listas para arrays do numpy
faces = [np.array(f, dtype=np.uint8) for f in faces] #lista de arrays uint8
labels = np.array(labels, dtype=np.int32) # array de inteiros 

# Cria o reconhecedor LBPH e treina com os dados
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# Salva o modelo treinado
recognizer.save("modelo_lbph.xml")

# Salva o dicionário de nomes
import pickle
with open("labels.pickle", "wb") as f:
    pickle.dump(label_map, f)

print("Treinamento concluído e modelo salvo!")