import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import base64
import time
import io
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import socketio

# --- Configuración (debe coincidir con el entrenamiento) ---
ACTIONS = ['hola', 'gracias'] # Asegúrate de que esto coincida con tu entrenamiento
SEQUENCE_LENGTH = 30
NUM_FEATURES = 21 * 3

# --- Cargar el Modelo ---
MODEL_PATH = 'modelo_lenguaje_senas.h5'
model = None # Inicializar a None
try:
    model = load_model(MODEL_PATH)
    print(f"Modelo cargado exitosamente desde: {MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    print("Asegúrate de que 'entrenamiento_modelo.py' se haya ejecutado y guardado el modelo.")
    print("El servidor continuará, pero las predicciones del modelo no funcionarán.")

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = None # Initialize hands to None

try:
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    print("MediaPipe Hands inicializado exitosamente.")
except Exception as e:
    print(f"ERROR: No se pudo inicializar MediaPipe Hands: {e}")
    print("Asegúrate de que tus instalaciones de TensorFlow y MediaPipe son compatibles.")
    # Consider adding os.environ['CUDA_VISIBLE_DEVICES'] = '-1' at the very top
    # if you suspect GPU-related issues.

# --- Socket.IO server for FastAPI ---
# sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
app = FastAPI()

# 2. CORS configured here for FastAPI HTTP routes (middleware)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], # Allow all origins during development
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Mount the Socket.IO server onto the FastAPI app
app.mount("/socket.io", socketio.ASGIApp(sio))


# --- Global state for each connected client ---
# We'll use dictionaries to store state for each sid (session ID)
# For a simple application, you might manage it directly in the websocket loop
# but this pattern is more robust for multiple clients.
client_states = {} # Store sequence_buffer, sentence_buffer, last_prediction_time per client

@sio.event
async def connect(sid, environ, auth):
    print(f'Cliente conectado: {sid}')
    client_states[sid] = {
        'sequence_buffer': [],
        'sentence_buffer': [],
        'last_prediction_time': time.time(),
    }
    await sio.emit('response', {'data': 'Conectado al servidor de señas'}, room=sid)

@sio.event
async def disconnect(sid):
    print(f'Cliente desconectado: {sid}')
    if sid in client_states:
        del client_states[sid]

@sio.event
async def video_frame(sid, data):
    # This event is for Socket.IO clients emitting 'video_frame'
    # Get state for this client
    state = client_states.get(sid)
    if not state:
        print(f"No state found for client {sid}. Disconnecting or re-initializing.")
        return

    sequence_buffer = state['sequence_buffer']
    sentence_buffer = state['sentence_buffer']
    last_prediction_time = state['last_prediction_time']

    if hands is None:
        print("MediaPipe Hands no está inicializado. No se puede procesar el frame.")
        await sio.emit('prediction', {'text': 'Error: MediaPipe no disponible'}, room=sid)
        return

    # Decode the base64 frame to an OpenCV image
    try:
        img_bytes = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return

    if frame is None:
        print("Frame is None after decoding. Skipping processing.")
        return

    # Process the frame with MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = None
    try:
        results = hands.process(image_rgb)
    except Exception as e:
        print(f"Error durante el procesamiento de MediaPipe: {e}")
        await sio.emit('prediction', {'text': 'Error: Procesando frame'}, room=sid)
        return
    image_rgb.flags.writeable = True

    keypoints = []
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
    else:
        keypoints = np.zeros(NUM_FEATURES).tolist()

    if len(keypoints) > NUM_FEATURES:
        keypoints = keypoints[:NUM_FEATURES]
    elif len(keypoints) < NUM_FEATURES:
        keypoints.extend([0.0] * (NUM_FEATURES - len(keypoints)))

    sequence_buffer.append(keypoints)
    sequence_buffer = sequence_buffer[-SEQUENCE_LENGTH:]

    current_time = time.time()
    prediction_interval = 0.5 # Defined locally for clarity, or get from config
    prediction_threshold = 0.8 # Defined locally for clarity, or get from config

    if model and len(sequence_buffer) == SEQUENCE_LENGTH and (current_time - last_prediction_time) > prediction_interval:
        input_data = np.expand_dims(sequence_buffer, axis=0)

        try:
            res = model.predict(input_data, verbose=0)[0]
        except Exception as e:
            print(f"Error durante la predicción del modelo: {e}")
            await sio.emit('prediction', {'text': 'Error de predicción'}, room=sid)
            state['last_prediction_time'] = current_time # Reset time to prevent error loop
            return

        predicted_action_idx = np.argmax(res)
        confidence = res[predicted_action_idx]

        if confidence > prediction_threshold:
            predicted_action = ACTIONS[predicted_action_idx]

            if len(sentence_buffer) == 0 or predicted_action != sentence_buffer[-1]:
                sentence_buffer.append(predicted_action)
                print(f"[{sid}] Predicción: {predicted_action} (Confianza: {confidence:.2f})")

            if len(sentence_buffer) > 5:
                sentence_buffer = sentence_buffer[-5:]

            await sio.emit('prediction', {'text': ' '.join(sentence_buffer)}, room=sid)
        else:
            pass # No new prediction if confidence is low

        state['last_prediction_time'] = current_time # Update state
    
    # Update the client's state dictionary
    client_states[sid]['sequence_buffer'] = sequence_buffer
    client_states[sid]['sentence_buffer'] = sentence_buffer
    client_states[sid]['last_prediction_time'] = last_prediction_time


# Optional: A simple HTTP endpoint for health check or root access
@app.get("/")
async def read_root():
    return {"message": "Servidor de Lenguaje de Señas con FastAPI y Socket.IO"}

# To run the FastAPI application:
# Use `uvicorn main:app --host 0.0.0.0 --port 5000 --ws websockets`
# (Replace 'main' with your Python file name if it's not main.py)
# if __name__ == '__main__':
#     print("Iniciando servidor FastAPI con Uvicorn en http://0.0.0.0:5000")
#     # Note: The '--ws websockets' flag is important for Socket.IO support
#     # when running from the command line, though it's often picked up automatically.
#     # For programmatic run, we need to pass the websocket library explicitly
#     uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True, ws="websockets")