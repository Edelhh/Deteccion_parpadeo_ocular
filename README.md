# Detección de parpadeo ocular usando técnica de detección de picos y filtro SG.
Este trabajo es una pequeña réplica de una investigación realizada en la literatura. Se recomienda leer el trabajo del autor para mayor conocimiento.

**Procedimiento:**
Se calcula la relación de aspecto del ojo o eye aspect ratio (EAR) de un video a 30 FPS. La medida EAR se guarda en un csv, donde se pasa a filtrar la señal mediante filtro Savitzky–Golay y se genera la predicción respectiva. Como librería principal para el cálculo de EAR se utiliza mediapipe.

**Trabajo de referencia:**
https://www.mdpi.com/2078-2489/9/4/93
