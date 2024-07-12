# Librerias
import cv2
import mediapipe as mp
import numpy as np
import csv
import pandas as pd
from collections import deque


def eye_aspect_ratio(coordinates):
    # Distancias euclidianas.
    distancia_p26 = round(np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[5])), 4)
    distancia_p35 = round(np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[4])), 4)
    distancia_p14 = round(np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3])), 4)
    return (distancia_p26 + distancia_p35) / (2 * distancia_p14), distancia_p26, distancia_p35, distancia_p14


def save_data_csv(data,nombre_data):
    df = pd.DataFrame(data, columns=data.keys())
    df.to_csv(nombre_data, sep = ";", header=True, index=True,index_label='frameID')


nombre_video = "video_02.avi"

nombre_data_eye = "video_02.avi"

cap = cv2.VideoCapture(nombre_data_eye)

mp_face_mesh = mp.solutions.face_mesh
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ear_total_data = []
contador_total = []

ear_left_eye_data = []
ear_right_eye_data = []

print(total_frames)
contador = 0

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        min_tracking_confidence=0.5,
        min_detection_confidence=0.5,
        max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        # frame = cv2.flip(frame, 1)
        contador = contador +1
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        coordinates_left_eye = []
        coordinates_right_eye = []

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:

                # Dibujo puntos ojo izquierdo
                for index in index_left_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_left_eye.append([x, y])
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), 1)
                    cv2.circle(frame, (x, y), 1, (128, 0, 255), 1)
                # Dibujo puntos ojo derecho
                for index in index_right_eye:
                    x = int(face_landmarks.landmark[index].x * width)
                    y = int(face_landmarks.landmark[index].y * height)
                    coordinates_right_eye.append([x, y])
                    cv2.circle(frame, (x, y), 2, (128, 0, 255), 1)
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

            ear_left_eye = eye_aspect_ratio(coordinates_left_eye)
            ear_right_eye = eye_aspect_ratio(coordinates_right_eye)
            print(contador)
            ear = (ear_left_eye + ear_right_eye) / 2

            ear_total_data.append(ear)
            ear_right_eye_data.append(ear_right_eye)
            ear_left_eye_data.append(ear_left_eye)
            contador_total.append(contador)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == ord('q'): break

eardata = {"Ear": ear_total_data, "EarDer": ear_right_eye_data, "EarIzq": ear_left_eye_data, "contador": contador_total }
save_data_csv(eardata,nombre_data_eye)

cap.release()
cv2.destroyAllWindows()
