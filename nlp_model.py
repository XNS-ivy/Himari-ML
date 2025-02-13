import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Dense
import numpy as np

# ✅ 1. Dataset sederhana (contoh kalimat + label)
texts = [
    "Saya sangat senang hari ini",
    "Hari ini cuaca cerah sekali",
    "Aku sangat kecewa dengan layanan ini",
    "Film itu sangat menyedihkan",
    "Saya bahagia bisa belajar TensorFlow",
    "Hari ini sangat buruk"
]
labels = [1, 1, 0, 0, 1, 0]  # 1 = Positif, 0 = Negatif

# ✅ 2. Preprocessing: Ubah teks ke vektor angka
vectorizer = TextVectorization(output_mode='int', output_sequence_length=5)
vectorizer.adapt(texts)
X = vectorizer(texts)  # Konversi teks ke angka
y = np.array(labels)

# ✅ 3. Bangun Model NLP dengan LSTM
model = keras.Sequential([
    Embedding(input_dim=1000, output_dim=16),  # Embedding untuk pemetaan kata ke vektor
    LSTM(32),  # LSTM untuk memahami konteks teks
    Dense(1, activation='sigmoid')  # Output: Sentimen positif (1) atau negatif (0)
])

# ✅ 4. Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ 5. Latih model (training)
model.fit(X, y, epochs=10, verbose=1)

# ✅ 6. Prediksi teks baru
def predict_sentiment(text):
    vectorized_text = vectorizer([text])
    prediction = model.predict(vectorized_text)
    sentiment = "Positif" if prediction[0][0] > 0.5 else "Negatif"
    return sentiment

# ✅ 7. Coba prediksi
test_text = "Saya sangat kecewa dengan produk ini"
print(f"Teks: {test_text}")
print(f"Prediksi Sentimen: {predict_sentiment(test_text)}")
