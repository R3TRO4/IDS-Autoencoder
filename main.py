import pandas as pd
import numpy as np
from pathlib import Path
import os


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
        full_df['Label'] = full_df['Label'].str.replace('', '-', regex=False)
        full_df['Label'] = full_df['Label'].str.replace('', '-', regex=False)

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
        print("[INFO] Pomijam preprocessing. Wczytuję dane z pliku...")

        try:
            full_df = pd.read_csv(output_path)
            print("[INFO] Dane wczytane pomyślnie!")
            return full_df
        except Exception as e:
            print(f"[BŁĄD] Plik istnieje, ale nie udało się go wczytać: {e}")
            # W razie błędu wczytywania, możemy spróbować przetworzyć dane od nowa
            print("[INFO] Próba ponownego przetworzenia danych surowych...")

    else:
        print(f"\n[INFO] Nie znaleziono pliku {output_filename} lub jest on pusty.")

    # Jeśli pliku nie ma (lub błąd wczytu), uruchamiamy pełny preprocessing
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


def display_summary(df):
    """
    Funkcja wyświetlająca podsumowanie wczytanego zbioru danych.
    """
    print("\n=== PODSUMOWANIE DANYCH ===")
    print(f"Wymiary zbioru: {df.shape}")
    print("Rozkład klas:")
    print(df['Label'].value_counts())


def main():
    # 1. Konfiguracja
    output_filename = "CIC-IDS2017_cleaned_combined.csv"

    # 2. Pobranie danych (z pliku lub przez preprocessing)
    full_df = load_or_process_data(output_filename)

    # 3. Wyświetlenie wyników (tylko jeśli dane zostały poprawnie załadowane)
    if full_df is not None:
        display_summary(full_df)


if __name__ == '__main__':
    main()