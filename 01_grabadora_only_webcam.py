import cv2
import mss
import numpy as np
import time
import pandas as pd
import datetime as dt


def generar_csv(datos,nombre_csv):
    datos = pd.DataFrame(datos, columns=datos.keys())
    datos.to_csv(nombre_csv,sep = ";", header=True, index=False)

# Parametros a configurar.
fps = 30 # Fotogramas por segundo
ancho_frame = 640
alto_frame = 480

# Ruta de carpeta para guardar datos de tiempo csv
ruta = '/Video/'

# Configuración de la captura de video de la cámara web
camara = cv2.VideoCapture(0)  # Índice 0 para la cámara web predeterminada
# Configuración del objeto de escritura de video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_camara = cv2.VideoWriter('video_camara.avi', fourcc, fps, (ancho_frame, alto_frame))

tiempos = []
datos_frame=[]
hora = []
tiempo_inicial = 0  # Tiempo de inicio de la grabación
contador_frame = 0

while True:
    tiempo_inicial_frame = time.time() # Tiempo de inicio de frame
    # Capturar video de la cámara web
    ret, frame = camara.read()
    if ret == False:
        break
    
    if contador_frame == 1:
        tiempo_inicial = tiempo_inicial_frame

    # Calculo tiempo de duración de una captura.
    tiempo_final_captura = time.time()  # Tiempo de finalización de la captura
    duracion_captura = tiempo_final_captura - tiempo_inicial_frame  # Duración de la captura en segundos
    duracion_captura = round(duracion_captura*1000,2)
    tiempos.append(str(duracion_captura))
    hora.append(dt.datetime.now())


    # Calculo FPS.
    contador_frame += 1.
    datos_frame.append(contador_frame)
    # Calcular el tiempo transcurrido
    tiempo_transcurrido = time.time() - tiempo_inicial
    # Calcular el FPS
    fps = contador_frame / tiempo_transcurrido
    # Imprimir el valor de FPS actual
    print(f"FPS: {fps:.2f}")
    # Mostrar el resultado en una ventana
    cv2.imshow("camara",frame)
    video_camara.write(frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

datos_tiempo = {"numero_frame":datos_frame ,"tiempo": tiempos, "hora": hora}
generar_csv(datos_tiempo,ruta+'tiempo.csv')

# Liberar recursos
video_camara.release()
camara.release()

cv2.destroyAllWindows()