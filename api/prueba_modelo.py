import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

# --- Configuración (debe coincidir con el entrenamiento) ---
ACTIONS = ['hola', 'gracias']
SEQUENCE_LENGTH = 30 # Longitud de la secuencia de fotogramas que espera el modelo
NUM_FEATURES = 21 * 3 # 21 keypoints * 3 coordenadas (x,y,z)

# --- Cargar el Modelo ---
MODEL_PATH = 'modelo_lenguaje_senas.h5'
try:
    model = load_model(MODEL_PATH)
    print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print("Asegúrate de que 'entrenamiento_modelo.py' se haya ejecutado y guardado el modelo.")
    exit()

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- Buffer para la secuencia de entrada del modelo ---
# Almacena los últimos SEQUENCE_LENGTH fotogramas de keypoints
sequence = []
sentence = [] # Para construir la frase de salida (ej. "hola gracias")
predictions = []
threshold = 0.8 # Umbral de confianza para aceptar una predicción

# --- Bucle de Captura y Predicción en Tiempo Real ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Iniciando reconocimiento de señas. Presiona 'q' para salir.")

last_prediction_time = time.time()
prediction_interval = 0.5 # Segundos entre predicciones para evitar saturación

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Voltear horizontalmente

    # Procesar el fotograma con MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
    else:
        # Si no se detectan manos, agregar ceros
        keypoints = np.zeros(NUM_FEATURES).tolist()

    # Asegurarse de que el array de keypoints tenga el tamaño correcto
    if len(keypoints) > NUM_FEATURES:
        keypoints = keypoints[:NUM_FEATURES]
    elif len(keypoints) < NUM_FEATURES:
        keypoints.extend([0.0] * (NUM_FEATURES - len(keypoints)))


    # Añadir los keypoints al buffer de la secuencia
    sequence.append(keypoints)
    sequence = sequence[-SEQUENCE_LENGTH:] # Mantener solo los últimos N fotogramas

    # Realizar predicción solo cuando la secuencia está llena y cada cierto intervalo
    if len(sequence) == SEQUENCE_LENGTH and (time.time() - last_prediction_time) > prediction_interval:
        # Reformatear la secuencia para que coincida con la entrada del modelo (1, SEQUENCE_LENGTH, NUM_FEATURES)
        input_data = np.expand_dims(sequence, axis=0)
        res = model.predict(input_data)[0] # Obtener las probabilidades de las clases

        predicted_action_idx = np.argmax(res)
        confidence = res[predicted_action_idx]

        if confidence > threshold:
            predicted_action = ACTIONS[predicted_action_idx]
            # Solo añadir la predicción a la "sentencia" si es una nueva acción
            # o si la misma acción es predicha con alta confianza consistentemente
            if len(sentence) == 0 or predicted_action != sentence[-1]:
                sentence.append(predicted_action)
                print(f"Predicción: {predicted_action} (Confianza: {confidence:.2f})")
            
            # Limitar la longitud de la sentencia para que no crezca indefinidamente
            if len(sentence) > 5:
                sentence = sentence[-5:]
        
        last_prediction_time = time.time() # Reiniciar el temporizador

    # Mostrar la predicción en pantalla
    display_text = ' '.join(sentence)
    cv2.putText(image, display_text, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Reconocimiento de Lenguaje de Señas', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Reconocimiento detenido.")