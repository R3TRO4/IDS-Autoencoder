from pathlib import Path
from preprocessing import *
from train_model import *
from evaluate_model import *
from sklearn.metrics import classification_report, confusion_matrix
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

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

    # Zmienna sterująca: czy mamy uruchomić preprocessing?
    should_process = False

    # 1. SPRAWDZENIE ISTNIENIA PLIKÓW TENSORÓW (.npy)
    if all(f.exists() for f in required_files):
        print(f"\n[INFO] Wszystkie pliki tensorów (.npy) już istnieją.")
        #choice = input("Czy chcesz przetworzyć dane od nowa? (t/n): ")
        choice = 'n'
        if choice.lower() == 't':
            should_process = True
        else:
            print("Pominięto tworzenie tensorów.")
            pass  # Wyjście z if'a
    else:
        # Jeśli plików brakuje, musimy przetwarzać
        print("\n[INFO] Brak kompletnych plików .npy. Rozpoczynam procedurę...")
        should_process = True

    # 2. WYKONANIE PREPROCESSINGU (jeśli flaga should_process == True)
    if should_process:
        # Najpierw wczytujemy CSV (funkcja load_or_process_data sama zapyta o cache CSV)
        full_df = load_or_process_data(output_filename)

        if full_df is not None:
            display_summary(full_df)
            # Tworzymy tensory
            make_tensors_and_save(full_df)
        else:
            print("Krytyczny błąd: Nie udało się załadować danych.")
            return

    # ==========================================
    # 1. WCZYTANIE DANYCH (TYLKO RAZ!)
    # ==========================================
    print("[INFO] Wczytywanie danych treningowych i testowych...")
    X_train, X_val = load_train_data()
    X_test, y_test = load_test_data()

    if X_train is None or X_val is None or X_test is None:
        print("BŁĄD: Nie udało się wczytać danych.")
        return

    input_dim = X_train.shape[1]

    # ==========================================
    # 2. DEFINICJA WARIANTÓW BADAWCZYCH (GŁĘBOKOŚĆ)
    # ==========================================
    warianty_modeli = {
        'Wariant 1 (1 warstwa)': 1,
        'Wariant 2 (3 warstwy)': 3,
        'Wariant 3 (5 warstw)': 5,
        'Wariant 4 (7 warstw)': 7,
        'Wariant 5 (9 warstw)': 9
    }

    liczba_iteracji = 5
    historia_wynikow = []

    print("\n" + "=" * 50)
    print(" ROZPOCZĘCIE ZAUTOMATYZOWANEGO EKSPERYMENTU (25 TRENINGÓW)")
    print("=" * 50 + "\n")

    # ==========================================
    # 3. GŁÓWNA PĘTLA BADAWCZA
    # ==========================================
    for nazwa_modelu, liczba_warstw in warianty_modeli.items():
        print(f"\n---> TESTOWANA ARCHITEKTURA: {nazwa_modelu} <---")

        for iteracja in range(1, liczba_iteracji + 1):
            print(f"  [Iteracja {iteracja}/{liczba_iteracji}] Rozpoczęcie treningu...")

            # Budowa modelu (przekazujemy liczbę warstw z pętli)
            model = build_autoencoder(input_dim, liczba_warstw)

            # TWORZYMY UNIKALNĄ NAZWĘ DLA KAŻDEGO MODELU
            nazwa_zapisywanego_modelu = f"model_glebokosc_{liczba_warstw}_iter_{iteracja}.keras"

            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                # Zapisujemy pod unikalną nazwą, żeby ich nie nadpisywać!
                ModelCheckpoint(nazwa_zapisywanego_modelu, monitor='val_loss', save_best_only=True, verbose=0)
            ]

            # Trening modelu
            history = model.fit(
                x=X_train, y=X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_val, X_val),
                callbacks=callbacks,
                verbose=0
            )

            # Wczytanie najlepszych wag z TEJ KONKRETNEJ iteracji
            model = load_model(nazwa_zapisywanego_modelu)

            # ==========================================
            # 4. TESTOWANIE I ZBIERANIE METRYK
            # ==========================================
            mse_errors = calculate_reconstruction_error(model, X_test)
            thresh_std, thresh_99, thresh_90, thresh_85 = find_threshold_statistics(mse_errors, y_test)

            aktualny_prog = thresh_99

            y_pred = (mse_errors > aktualny_prog).astype(int)

            raport = classification_report(y_test, y_pred, target_names=['Normalny', 'Atak'], output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            TN, FP, FN, TP = cm.ravel()

            precyzja = raport['Atak']['precision']
            czulosc = raport['Atak']['recall']
            f1 = raport['Atak']['f1-score']

            pojedynczy_wynik = {
                'Model': nazwa_modelu,
                'Iteracja': iteracja,
                'Prog (99 perc.)': aktualny_prog,
                'Precyzja': precyzja,
                'Czulosc (Recall)': czulosc,
                'F1-Score': f1,
                'Pominiete Ataki (FN)': FN
            }
            historia_wynikow.append(pojedynczy_wynik)

            print(f"    -> Zakończono! F1 = {f1:.4f}, FN = {FN}")

            # Zmienione nazwy plików wykresów, by się nie pomieszały z Wąskim Gardłem!
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normalny', 'Atak'],
                        yticklabels=['Normalny', 'Atak'])
            plt.title(f'Macierz Pomyłek - {nazwa_modelu} (Iter: {iteracja})')
            plt.savefig(f"cm_glebokosc_{liczba_warstw}_iter_{iteracja}.png", bbox_inches='tight')
            plt.close()

            tf.keras.backend.clear_session()

    # ==========================================
    # 5. PODSUMOWANIE I UŚREDNIANIE (PANDAS)
    # ==========================================
    df_wyniki = pd.DataFrame(historia_wynikow)
    df_wyniki.to_csv('pelne_wyniki_eksperyment_glebokosc.csv', index=False)

    print("\n\n" + "=" * 50)
    print(" SUROWE WYNIKI (25 ITERACJI):")
    print(df_wyniki.to_string())

    # Agregacja wyników (średnia i odchylenie standardowe)
    podsumowanie = df_wyniki.groupby('Model').agg({
        'F1-Score': ['mean', 'std'],
        'Czulosc (Recall)': ['mean', 'std'],
        'Pominiete Ataki (FN)': ['mean', 'std'],
        'Prog (99 perc.)': ['mean']
    }).reset_index()

    print("\n" + "=" * 70)
    print(" OSTATECZNE WYNIKI DLA PRACY DYPLOMOWEJ (ŚREDNIE Z 5 ITERACJI):")
    print("=" * 70)
    print(podsumowanie.to_string())

    podsumowanie.to_csv('srednie_wyniki_glebokosc.csv', index=False)
    print("\n[INFO] Zapisano pliki .csv z wynikami. Eksperyment zakończony pomyślnie!")

if __name__ == '__main__':
    main()