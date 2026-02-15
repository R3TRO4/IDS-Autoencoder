import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import os

# Ustawienie losowości dla powtarzalności wyników
tf.random.set_seed(4)
np.random.seed(4)

# --- 3. ETAP TRENOWANIA (ETAP 3) ---
def load_train_data():
    # Wczytanie przygotowanych wcześniej plików .npy
    print("[INFO] Wczytywanie danych treningowych...")

    if not os.path.exists('data_train.npy'):
        print("[BŁĄD] Nie znaleziono pliku data_train.npy! Uruchom najpierw main.py")
        return None, None, None

    X_train = np.load('data_train.npy')
    X_val = np.load('data_val.npy')
    # Zbiór testowy zostaniew czytany dopiero w pliku do ewaluacji, tu jest niepotrzebny

    print(f"Dane wczytane:")
    print(f" -> Treningowe: {X_train.shape}")
    print(f" -> Walidacyjne: {X_val.shape}")

    return X_train, X_val

def build_autoencoder(input_dim):
    """
    Definiuje architekturę sieci neuronowej.
    Miejsce do eksperymentowania z liczbą warstw i neuronów.
    :param input_dim: Liczba cech (kolumn) z danych
    :return: autoencoder (model)
    """

    # 1. WARSTWA WEJŚCIOWA
    input_layer = Input(shape=(input_dim,))

    # 2. ENCODER (Kompresja)
    # Zmniejszamy wymiarowość, szukając ukrytych wzorców
    encoder = Dense(64, activation='relu')(input_layer)
    encoder = Dense(32, activation='relu')(encoder)

    # 3. BOTTLENECK (Najwęższe gardło)
    # To tutaj dzieje się kompresja wiedzy o całym ruchu do zaledwie np. 16 liczb
    bottleneck = Dense(16, activation='relu', name='bottleneck')(encoder)

    # 4. DECODER (Rekonstrukcja)
    # Odbicie lustrzane Encodera - próba odtworzenia oryginału
    decoder = Dense(32, activation='relu')(bottleneck)
    decoder = Dense(64, activation='relu')(decoder)

    # 5. WARSTWA WYJŚCIOWA
    # Ważne: Aktywacja 'sigmoid', ponieważ nasze dane są znormalizowane do [0, 1]
    # Sigmoid zwraca wartości właśnie w tym przedziale.
    output_layer = Dense(input_dim, activation='sigmoid')(decoder)

    # Złożenie modelu
    autoencoder = Model(inputs=input_layer, outputs=output_layer)

    # Kompilacja
    # Optimizer 'adam' to standard. Loss 'mse" (błąd średniokwadratowy)
    # jest idealny do mierzenia jakości rekonstrukcji.
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

def plot_history(history):
    # Rysuje wykres funkcji straty
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b', label='Strata treningowa (Training Loss)')
    plt.plot(epochs, val_loss, 'r', label='Strata walidacyjna (Validation Loss)')
    plt.title('Przebieg uczenia Autoencodera')
    plt.xlabel('Epoki')
    plt.ylabel('Błąd rekonstrukcji (MSE)')
    plt.legend()
    plt.grid(True)

    # Zapis wykresu do pliku
    plt.savefig("training_history.png")
    print("[INFO] Wykres zapisano jako training_history.png")
    plt.show()