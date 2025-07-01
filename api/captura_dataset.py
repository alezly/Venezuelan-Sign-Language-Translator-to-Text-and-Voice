import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import time

# --- Configuración ---
DATA_DIR = 'data_sign_language' # Directorio principal para guardar los datos
ACTIONS = ['gracias'] # EJEMPLO: Lista de señas a capturar
# Asegúrate de que estas sean señas que puedas realizar tanto estáticas como dinámicas si aplica.
# 'hola' puede ser dinámica, 'yo' puede ser estática, 'comer' puede ser dinámica.
NO_SEQUENCES = 8 # Número de secuencias (ejemplos) por cada seña
SEQUENCE_LENGTH = 50 # Número de fotogramas por cada secuencia (para señas dinámicas o como mínimo para estáticas)
# Un SEQUENCE_LENGTH de 30 fotogramas a 30 FPS son 1 segundo de video.
# Para señas estáticas, MediaPipe seguirá detectando los keypoints, y el modelo aprenderá que la posición es constante.

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Crea la estructura de directorios si no existe
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        try:
            os.makedirs(os.path.join(DATA_DIR, action, str(sequence)))
        except FileExistsError:
            pass # Directorio ya existe

print("Preparado para capturar el dataset.")
print(f"Directorio de salida: {DATA_DIR}")
print(f"Señas a capturar: {ACTIONS}")
print(f"Número de secuencias por seña: {NO_SEQUENCES}")
print(f"Longitud de cada secuencia (fotogramas): {SEQUENCE_LENGTH}")

# --- Bucle de Captura ---
cap = cv2.VideoCapture(0) # 0 para la webcam predeterminada

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

for action_idx, action in enumerate(ACTIONS):
    for sequence_idx in range(NO_SEQUENCES):
        # Pausa para preparación
        cv2.waitKey(2000) # Espera 2 segundos antes de iniciar la captura de cada secuencia

        # Mensaje en consola y en la ventana
        print(f"Capturando {action} - Secuencia {sequence_idx + 1}/{NO_SEQUENCES}. Prepárate...")
        frame_display_text = f"Preparate para: {action} ({sequence_idx + 1}/{NO_SEQUENCES})"
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1) # Voltear horizontalmente para vista de espejo
            cv2.putText(frame, frame_display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Captura de Señas', frame)
            cv2.waitKey(1) # Pequeña espera para mostrar el texto

        # Bucle para capturar la secuencia de fotogramas
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1) # Voltear horizontalmente

            # Procesar el fotograma con MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False # Optimización
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extraer puntos clave
            keypoints = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar los landmarks y conexiones
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # Extraer coordenadas para el dataset
                    for landmark in hand_landmarks.landmark:
                        keypoints.extend([landmark.x, landmark.y, landmark.z])
            else:
                # Si no se detectan manos, guardar un array de ceros para ese fotograma
                # Esto es importante para mantener la longitud de la secuencia
                keypoints = np.zeros(21 * 3).tolist() # 21 landmarks * 3 coordenadas (x, y, z)

            # Asegurarse de que el array de keypoints tenga el tamaño correcto
            # Si se detectan dos manos, o ninguna, el tamaño puede variar.
            # Para este ejemplo, solo usaremos la primera mano detectada o ceros.
            if len(keypoints) > (21 * 3): # Si se detectaron dos manos, solo tomamos la primera
                keypoints = keypoints[:(21*3)]
            elif len(keypoints) < (21 * 3): # Si no se detectó ninguna o menos de 21, rellenamos con ceros
                keypoints.extend([0.0] * ((21 * 3) - len(keypoints)))

            # Guardar los keypoints en el archivo
            npy_path = os.path.join(DATA_DIR, action, str(sequence_idx), str(frame_num))
            np.save(npy_path, np.array(keypoints))

            # Mostrar el fotograma con los landmarks
            cv2.putText(image, f"Capturando: {action} ({sequence_idx + 1}/{NO_SEQUENCES}) - Frame: {frame_num + 1}/{SEQUENCE_LENGTH}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Captura de Señas', image)

            # Esperar un poco para asegurar que la cámara tiene tiempo de capturar bien
            # Y para que el usuario pueda mantener la postura o el movimiento
            cv2.waitKey(1) # Espera 1ms

        print(f"Secuencia {sequence_idx + 1} de {action} capturada.")

cv2.destroyAllWindows()
cap.release()
hands.close()
print("Dataset capturado exitosamente.")