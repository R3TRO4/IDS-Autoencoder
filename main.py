import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# --- 1. FUNKCJE POMOCNICZE DO CZYSZCZENIA (ETAP 1) ---
def clean_chunk(df):
    """
    Funkcja pomocnicza czyszcząca pojedynczy fragment danych.
    """
    print(f"\n[INFO] Rozmiar przed czyszczeniem: {df.shape}")

    # ETAP 1: Usunięcie spacji z nazw kolumn.
    df.columns = df.columns.str.strip()

    # Etap 3: Obsługa NaN i Infinity
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"[INFO] Rozmiar po czyszczeniu: {df.shape}")
    return df


def preprocessing_data():
    """
    Główna funkcja orkiestrująca cały proces przygotowania danych (raw data).
    ZWRACA: DataFrame z połączonymi danymi lub None w przypadku błędu.
    """
    print("Welcome to Autoencoder IDS Project - PREPROCESSING MODE")

    data_dir = Path("CIC-IDS2017")
    data_frames = []

    print(f"[INFO] Szukam plików w: {data_dir.absolute()}")
    csv_files = list(data_dir.glob('*.csv'))

    if not csv_files:
        print("BŁĄD: Nie znaleziono żadnych plików .csv!")
        return None

    for file_path in csv_files:
        print(f"\n---> Przetwarzanie pliku: {file_path.name}")
        try:
            df_part = pd.read_csv(file_path)
            df_part = clean_chunk(df_part)
            data_frames.append(df_part)
            del df_part
        except Exception as e:
            print(f"Błąd przy przetwarzaniu {file_path.name}: {e}")

    if data_frames:
        print("\n[INFO] Łączenie wszystkich plików w jeden zbiór danych...")
        full_df = pd.concat(data_frames, ignore_index=True)

        print("=== FINALNY ZBIÓR DANYCH (Skompilowany) ===")
        print(f"Łączna liczba wierszy: {full_df.shape[0]}")

        print("\n[INFO] Naprawianie nazw etykiet...")
        full_df['Label'] = full_df['Label'].str.replace('�', '-', regex=False)
        full_df['Label'] = full_df['Label'].str.replace('�', '-', regex=False)

        return full_df
    else:
        print("Nie udało się utworzyć zbioru danych.")
        return None


def load_or_process_data(output_filename):
    """
    Funkcja zarządzająca cache'owaniem danych.
    Sprawdza, czy istnieje gotowy plik. Jeśli tak - wczytuje go.
    Jeśli nie - uruchamia preprocessing i zapisuje wynik.
    """
    output_path = Path(output_filename)

    # Sprawdzamy:
    # 1. Czy plik istnieje
    # 2. Czy jego rozmiar jest większy od 0 bajtów
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"\n[INFO] Wykryto gotowy plik z danymi: {output_filename}")
        choice = input("Czy chcesz skorzystać z danych z pliku? (t/n): ")
        if (choice.lower() == 't'):
            try:
                print("[INFO] Pomijam preprocessing. Wczytuję dane z pliku...")
                full_df = pd.read_csv(output_path)
                print("[INFO] Dane wczytane pomyślnie!")
                return full_df
            except Exception as e:
                print(f"[BŁĄD] Plik istnieje, ale nie udało się go wczytać: {e}")
                # W razie błędu wczytywania, możemy spróbować przetworzyć dane od nowa
                print("[INFO] Próba ponownego przetworzenia danych surowych...")
        elif (choice.lower() != 't'):
            # Uruchamiamy pełny preprocessing
            print("[INFO] Rozpoczynam pełny preprocessing...")
            full_df = preprocessing_data()
            # Jeśli preprocessing się udał (dostaliśmy dane), zapisujemy je do pliku
            if full_df is not None:
                print(f"\n[INFO] Zapisywanie wyników do pliku: {output_filename}...")
                full_df.to_csv(output_path, index=False)
                print("[INFO] Zapis zakończony sukcesem.")
                return full_df
            else:
                print("[BŁĄD] Preprocessing nie zwrócił danych. Plik nie został utworzony.")
                return None


# --- 2. PRZYGOTOWANIE POD TRENING (ETAP 2) ---
def make_tensors_and_save(df):
    """
    Metoda przygotowuje dane w podziale na 3 zbiory:
    1. TRENINGOWY (Czysty Benign) - do uczenia wag.
    2. WALIDACYJNY (Czysty Benign) - do Early Stopping (zapobiega przeuczeniu)
    3. TESTOWY (Mieszanka) - do finalnej oceny detekcji anomalii
    """

    print("\n[INFO] --- PRZYGOTOWANIE DANYCH (TRAIN / VAL / TEST) ---")
    # Krok A: Przygotowanie kolumn
    # Stworzenie kopii, aby nie psuć oryginału
    df = df.copy()

    # Tworzymy kolumnę pomocniczą (0 = Benign, 1 = Attack)
    df['Label_Binary'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    # Krok B: Rozdzielenie ruchu Normalnego od Ataków
    print("[1/5] Separacja ruchu Normalnego i Ataków")
    df_benign = df[df['Label_Binary'] == 0]
    df_attack = df[df['Label_Binary'] == 1]

    # Usuwamy erykiety z danych wejściowych (sieć ma widzieć tylko cechy sieciowe)
    cols_to_drop = ['Label', 'Label_Binary']
    X_benign = df_benign.drop(columns=cols_to_drop, errors='ignore')
    X_attack = df_attack.drop(columns=cols_to_drop, errors='ignore')

    # Krok C: Podział ruchu NORMALNEGO na 3 części
    # Strategia: 70% Trening, 15% Walidacja, 15% Test
    print("[2/5] Dzielenie ruchu normalnego na Train/Val/Test...")

    # Najpierw wydzielam zbiór treningowy (70%) i resztę (30%)
    X_train, X_temp = train_test_split(X_benign, test_size=0.30, random_state=42)

    # Teraz dzielę resztę (30%) na pół -> 15% Walidacja, 15% Test
    X_val, X_test_benign = train_test_split(X_temp, test_size=0.50, random_state=42)

    print(f"      Rozmiar Train (Benign): {X_train.shape[0]} (do nauki)")
    print(f"      Rozmiar Val   (Benign): {X_val.shape[0]}   (do kontroli)")
    print(f"      Rozmiar Test  (Benign): {X_test_benign.shape[0]} (do weryfikacji FPR)")
    print(f"      Rozmiar Attack (All):   {X_attack.shape[0]}   (do weryfikacji wykrywania)")

    # Krok D: Normalizacja (MinMaxScaler)
    print("[3/5] Skalowanie danych (MinMaxScaler)...")
    scaler = MinMaxScaler()

    # UWAGA: Skaler uczymy TYLKO na zbiorze treningowym!
    # Walidacyjny i Testowy muszą być przeskalowane względem wiedzy ze zbioru treningowego.
    X_train_scaled = scaler.fit_transform(X_train)

    # Transformujemy resztę
    X_val_scaled = scaler.transform(X_val)
    X_test_benign_scaled = scaler.transform(X_test_benign)
    X_attack_scaled = scaler.transform(X_attack)

    # Krok E: Złożenie finalnego zbioru testowego
    # Zbiór testowy zawiera: 15% czystego ruchu ORAZ wszystkie ataki
    X_test_final = np.concatenate([X_test_benign_scaled, X_attack_scaled])

    # Etykiety dla testu (0 = Benign, 1 = Attack) - potrzebne do wyliczenia skuteczności
    y_test_benign = np.zeros(len(X_test_benign_scaled))
    y_test_attack = np.ones(len(X_attack_scaled))
    y_test_final = np.concatenate([y_test_benign, y_test_attack])

    # Krok F: Zapis do plików
    print("[4/5] Zapisywanie gotowych macierzy (.npy)...")

    # 1. Zbiór Treningowy
    np.save("data_train.npy", X_train_scaled)

    # 2. Zbiór Walidacyjny
    np.save("data_val.npy", X_val_scaled)

    # 3. Zbiór Testowy (Cechy + Etykiety)
    np.save("data_test.npy", X_test_final)
    np.save("data_test_labels.npy", y_test_final)

    # 4. Scaler
    joblib.dump(scaler, "scaler_data.save")

    print("\n[5/5] === SUKCES ===")
    print(f"Pliki gotowe do użycia:")
    print(f" -> data_train.npy (Benign): {X_train_scaled.shape}")
    print(f" -> data_val.npy   (Benign): {X_val_scaled.shape}")
    print(f" -> data_test.npy  (Mixed):  {X_test_final.shape}")
    print(f" -> data_test_labels.npy:    {y_test_final.shape}")

def display_summary(df):
    """
    Funkcja wyświetlająca podsumowanie wczytanego zbioru danych.
    """
    print("\n=== PODSUMOWANIE DANYCH ===")
    print(f"Wymiary zbioru: {df.shape}")
    print("Rozkład klas:")
    print(df['Label'].value_counts())


def main():
    # Nazwa pliku, który może zawierać dane
    output_filename = "CIC-IDS2017_cleaned_combined.csv"

    # Lista plików, które muszą istnieć, żeby uznać, że "wszystko gotowe"
    required_files = [
        Path("data_train.npy"),
        Path("data_val.npy"),
        Path("data_test.npy"),
        Path("data_test_labels.npy")
    ]

    # Zmienna sterująca: czy mamy uruchomić przetwarzanie?
    should_process = False

    # 1. SPRAWDZENIE ISTNIENIA PLIKÓW TENSORÓW (.npy)
    if all(f.exists() for f in required_files):
        print(f"\n[INFO] Wszystkie pliki tensorów (.npy) już istnieją.")
        choice = input("Czy chcesz przetworzyć dane od nowa? (t/n): ")

        if choice.lower() == 't':
            should_process = True
        else:
            print("Pominięto tworzenie tensorów.")
            pass  # Wyjście z funkcji main (i programu)
    else:
        # Jeśli plików brakuje, musimy przetwarzać
        print("\n[INFO] Brak kompletnych plików .npy. Rozpoczynam procedurę...")
        should_process = True

    # 2. WYKONANIE LOGIKI (jeśli flaga should_process == True)
    if should_process:
        # Najpierw wczytujemy CSV (funkcja load_or_process_data sama zapyta o cache CSV)
        full_df = load_or_process_data(output_filename)

        if full_df is not None:
            display_summary(full_df)
            # Tworzymy tensory
            make_tensors_and_save(full_df)
        else:
            print("Krytyczny błąd: Nie udało się załadować danych.")


if __name__ == '__main__':
    main()