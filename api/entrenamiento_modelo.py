import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuración del Dataset (debe coincidir con captura_dataset.py) ---
DATA_DIR = 'data_sign_language'
ACTIONS = ['hola', 'gracias']
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
# Número de características por fotograma (21 keypoints * 3 coordenadas x,y,z)
NUM_FEATURES = 21 * 3

# --- 1. Cargar el Dataset ---
print("Cargando el dataset...")
sequences, labels = [], []
for action_idx, action in enumerate(ACTIONS):
    for sequence_idx in range(NO_SEQUENCES):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            try:
                # Cargar el archivo .npy para cada fotograma
                res = np.load(os.path.join(DATA_DIR, action, str(sequence_idx), f"{frame_num}.npy"))
                # Asegurarse de que el array tenga la forma correcta
                if res.shape[0] != NUM_FEATURES:
                    print(f"Advertencia: Archivo {os.path.join(DATA_DIR, action, str(sequence_idx), f'{frame_num}.npy')} tiene forma {res.shape}, se esperaba {NUM_FEATURES}. Rellenando/truncando.")
                    # Rellenar con ceros o truncar si el tamaño no coincide
                    if res.shape[0] < NUM_FEATURES:
                        res = np.pad(res, (0, NUM_FEATURES - res.shape[0]), 'constant')
                    else:
                        res = res[:NUM_FEATURES]

                window.append(res)
            except FileNotFoundError:
                print(f"Advertencia: Archivo no encontrado {os.path.join(DATA_DIR, action, str(sequence_idx), f'{frame_num}.npy')}. Saltando o rellenando con ceros.")
                window.append(np.zeros(NUM_FEATURES)) # Rellenar con ceros si falta un archivo
        sequences.append(window)
        labels.append(action_idx)

# Convertir a arrays de NumPy
X = np.array(sequences)
y = to_categorical(np.array(labels)).astype(int)

print(f"Forma de X (secuencias): {X.shape}") # (NO_SEQUENCES * len(ACTIONS), SEQUENCE_LENGTH, NUM_FEATURES)
print(f"Forma de y (etiquetas): {y.shape}")   # (NO_SEQUENCES * len(ACTIONS), len(ACTIONS))
print("Dataset cargado.")

# --- 2. Dividir el Dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

# --- 3. Construir el Modelo LSTM (o Híbrido CNN-LSTM si usaras imágenes directas) ---
# Aquí usamos solo LSTM porque la extracción de características (keypoints) ya se hizo.
# Una capa Bidireccional LSTM puede mejorar el rendimiento al procesar la secuencia en ambas direcciones.

model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu'), input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(256, return_sequences=False, activation='relu')))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(ACTIONS), activation='softmax')) # Capa de salida con softmax para clasificación multiclase

print("\nModelo construido:")
model.summary()

# --- 4. Compilar el Modelo ---
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 5. Callbacks para el Entrenamiento ---
# TensorBoard para visualizar el progreso
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Early Stopping para detener el entrenamiento si la validación no mejora
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Reduce learning rate cuando la métrica no mejora
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)

# --- 6. Entrenar el Modelo ---
print("\nEntrenando el modelo...")
history = model.fit(X_train, y_train,
                    epochs=500, # Puedes ajustar las épocas. Early Stopping lo detendrá si es suficiente.
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[tb_callback, early_stopping, reduce_lr])

print("\nEntrenamiento finalizado.")

# --- 7. Evaluar el Modelo ---
print("\nEvaluando el modelo...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Pérdida en el conjunto de prueba: {loss:.4f}")
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

# Métricas más detalladas
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nMatriz de Confusión por Clase:")
cm = multilabel_confusion_matrix(y_true_classes, y_pred_classes)
# Para una mejor visualización, puedes usar sklearn.metrics.confusion_matrix y seaborn
# import sklearn.metrics
# conf_matrix = sklearn.metrics.confusion_matrix(y_true_classes, y_pred_classes)
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=ACTIONS, yticklabels=ACTIONS)
# plt.xlabel('Predicción')
# plt.ylabel('Verdadero')
# plt.title('Matriz de Confusión')
# plt.show()


print(f"Precisión general (usando sklearn): {accuracy_score(y_true_classes, y_pred_classes):.4f}")

# Visualizar el historial de entrenamiento
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.tight_layout()
plt.show()

# --- 8. Guardar el Modelo ---
model_save_path = 'modelo_lenguaje_senas.h5'
model.save(model_save_path)
print(f"Modelo guardado en: {model_save_path}")