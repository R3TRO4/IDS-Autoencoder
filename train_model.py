import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from setuptools.sandbox import save_path
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
        return None, None

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
    #encoder = Dense(512, activation='gelu')(input_layer)
    encoder = Dense(256, activation='gelu')(input_layer)
    encoder = Dense(128, activation='gelu')(encoder)
    encoder = Dense(64, activation='gelu')(encoder)
    encoder = Dense(32, activation='gelu')(encoder)

    # 3. BOTTLENECK (Najwęższe gardło)
    # To tutaj dzieje się kompresja wiedzy o całym ruchu do zaledwie np. 16 liczb
    bottleneck = Dense(16, activation='gelu', name='bottleneck')(encoder)

    # 4. DECODER (Rekonstrukcja)
    # Odbicie lustrzane Encodera - próba odtworzenia oryginału
    decoder = Dense(32, activation='gelu')(bottleneck)
    decoder = Dense(64, activation='gelu')(decoder)
    decoder = Dense(128, activation='gelu')(decoder)
    decoder = Dense(256, activation='gelu')(decoder)
    #decoder = Dense(512, activation='gelu')(decoder)

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

def plot_history(history, save_path=None):
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

    # Automatyczny zapis jeśli podano ścieżkę
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300) # Wysoka jakość do druku
        print(f"[INFO] Wykres historii uczenia zapisano jako: {save_path}")
        plt.close() # Zamknij w tle, nie blokuj pętli
    else:
        plt.show()