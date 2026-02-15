import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

def load_test_data():
    print("[INFO] Wczytywanie danych testowych...")
    X_test = np.load("data_test.npy")
    y_test = np.load("data_test_labels.npy")
    return X_test, y_test


def calculate_reconstruction_error(model, X):
    # Oblicza błąd MSE dla każdej próbki oddzielnie.

    print("[INFO] Generowanie rekonstrukcji...")
    reconstructions = model.predict(X)

    # MSE dla każdego wiersza: średnia z kwadratów różnic
    # axis=1 oznacza, że liczymy średnią po cechach dla każdego wiersza
    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
    return mse


def find_threshold_statistics(mse_errors, y_true):
    # Pomocnicza funkcja do analizy progu.
    # Patrzymy, jaki błąd mają próbki normalne (y=0).

    normal_errors = mse_errors[y_true == 0]
    attack_errors = mse_errors[y_true == 1]

    print("\n=== STATYSTYKI BŁĘDU REKONSTRUKCJI ===")
    print(f"Średni błąd (Normal): {np.mean(normal_errors):.6f}")
    print(f"Max błąd (Normal):    {np.max(normal_errors):.6f}")
    print(f"Std błąd (Normal):    {np.std(normal_errors):.6f}")
    print("-" * 30)
    print(f"Średni błąd (Atak):   {np.mean(attack_errors):.6f}")
    print(f"Min błąd (Atak):      {np.min(attack_errors):.6f}")

    # Strategia 1: Średnia + 3 odchylenia standardowe (Reguła 3 sigma)
    threshold_std = np.mean(normal_errors) + 3 * np.std(normal_errors)

    # Strategia 2: 99. percentyl (odcinamy 1% najtrudniejszych normalnych próbek)
    threshold_99 = np.percentile(normal_errors, 99)

    return threshold_std, threshold_99


def plot_error_distribution(mse_errors, y_true, threshold):
    # Rysuje histogram błędów - kluczowy wykres do pracy mgr!
    # Pokazuje, jak bardzo różnią się błędy dla ataków i ruchu normalnego.
    plt.figure(figsize=(12, 6))

    # Rysujemy histogram dla ruchu normalnego
    sns.histplot(mse_errors[y_true == 0], bins=50, kde=True, color='green', label='Ruch Normalny', alpha=0.6)

    # Rysujemy histogram dla ataków
    sns.histplot(mse_errors[y_true == 1], bins=50, kde=True, color='red', label='Atak', alpha=0.6)

    # Rysujemy linię progu
    plt.axvline(threshold, color='blue', linestyle='--', label=f'Próg odcięcia ({threshold:.4f})')

    plt.title('Rozkład błędu rekonstrukcji (Normal vs Atak)')
    plt.xlabel('Błąd rekonstrukcji (MSE)')
    plt.ylabel('Liczba próbek (Skala logarytmiczna)')
    plt.yscale('log')  # Skala logarytmiczna, bo próbek benign jest dużo więcej
    plt.legend()

    plt.savefig("error_distribution.png")
    print("[INFO] Zapisano wykres rozkładu błędów: error_distribution.png")
    plt.show()