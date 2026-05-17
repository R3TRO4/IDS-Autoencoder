import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from setuptools.sandbox import save_path

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

    # Strategia 3: 90. percentyl (odcinamy 10% najtrudniejszych normalnych próbek)
    threshold_90 = np.percentile(normal_errors, 90)

    # Strategia 4: 85. percentyl (odcinamy 15% najtrudniejszych normalnych próbek)
    threshold_85 = np.percentile(normal_errors, 85)

    return threshold_std, threshold_99, threshold_90, threshold_85


def plot_error_distribution(mse_errors, y_true, threshold, save_path=None):
    # Rysuje histogram błędów - zoptymalizowany dla przejrzystości klas
    plt.figure(figsize=(12, 6))

    # Wyznaczamy limit osi X (robimy "zoom" na kluczowy obszar)
    # Ignorujemy ekstremalne outliery, żeby zobaczyć separację.
    max_x = threshold * 15

    # Rysujemy histogramy (KDE=False naprawia problem osi Y)
    sns.histplot(mse_errors[y_true == 0], bins=100, kde=False, color='green',
                 label='Ruch Normalny', alpha=0.6, binrange=(0, max_x))

    sns.histplot(mse_errors[y_true == 1], bins=100, kde=False, color='red',
                 label='Atak', alpha=0.6, binrange=(0, max_x))

    # Rysujemy linię progu
    plt.axvline(threshold, color='blue', linestyle='--',
                label=f'Próg odcięcia ({threshold:.4f})')

    plt.title('Rozkład błędu rekonstrukcji (Normal vs Atak)')
    plt.xlabel('Błąd rekonstrukcji (MSE)')
    plt.ylabel('Liczba próbek (Skala logarytmiczna)')

    plt.yscale('log')
    plt.xlim(0, max_x)  # Wymuszamy cięcie osi X

    plt.legend()
    plt.tight_layout()

    # Zapis pod unikalną nazwą (jeśli została podana) w wysokiej jakości dpi=300
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Zapisano wykres rozkładu błędów: {save_path} (Limit X: {max_x:.4f})")
        plt.close()  # Zamknij okienko w tle, żeby skrypt się nie zatrzymał
    else:
        plt.show()