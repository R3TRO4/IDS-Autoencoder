from pathlib import Path
from preprocessing import *
from train_model import *
from evaluate_model import *
import random

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
        choice = input("Czy chcesz przetworzyć dane od nowa? (t/n): ")

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

    # 3. ZBUDOWANIE I TRENING AUTOENCODERA
    # Wczytanie danych
    X_train, X_val = load_train_data()

    if X_train is None or X_val is None:
        return

    # Pobieramy liczbę cech (kolumn) dynamicznie z danych
    input_dim = X_train.shape[1]

    # Budowa modelu
    model = build_autoencoder(input_dim)
    model.summary() # Wyświetla tabelkę z architekturą w konsoli

    # Konfiguracja Callbacków (mechanizmów kontrolnych)
    callbacks = [
        # EarlyStopping: Jeśli val_loss nie spadnie przez 5 epok, przerwij uczenie.
        # Ten machanizm zapobiega przeuczeniu (overfitting)
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        # ModelCheckpoint: Zapisuj model tylko wtedy, gdy jest najlepszy (najmniejszy błąd)
        ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=0)
    ]

    # Trenowanie modelu
    print("\n[INFO] Rozpoczęcie treningu sieci...")
    # UWAGA: W Autoencoderze X (wejście) jest równe Y (oczekiwane wyjście)!
    # Dkategi oidaheny x=X_train, y=X_train
    history = model.fit(
        x=X_train,
        y=X_train,
        epochs=50,          # Maksymalna liczba epok (EarlyStopping i tak przerwie wcześniej)
        batch_size=256,     # Ile próbek na raz mieli karta graficzna/CPU
        shuffle=True,       # Mieszanie danych w każdej epoce
        validation_data=(X_val, X_val), # Sprawdzanie jakości na zbiorze walidacyjnym
        callbacks=callbacks
    )

    # Zapisz finalny model i wykres
    print("[INFO] Trening zakończony.")
    plot_history(history)

    # Model jest już zapisany przez Checkpoint jako 'best_model.keras'
    print("Najlepszy model zapisano jako: best_model.keras")


    # 4. TESTOWANIE MODELU
    # Załadowanie modelu
    model_path = 'best_model.keras'
    print("[INFO] Wczytanie modelu: {model_path}")
    try:
        model = load_model(model_path)
    except:
        print("BŁĄD: Nie znaleziono modelu 'best_model.keras'. Uruchom najpierw train_model.py!")
        return

    X_test, y_test = load_test_data()

    # Obliczenie błędów
    mse_errors = calculate_reconstruction_error(model, X_test)

    # Wyznaczenie progu
    # To jest serce Twojego problemu badawczego ("dobór progu błędu")
    thresh_std, thresh_99, thresh_90, thresh_85 = find_threshold_statistics(mse_errors, y_test)

    print(f"\nSugerowane progi:")
    print(f" -> Metoda statystyczna (Mean + 3*Std): {thresh_std:.6f}")
    print(f" -> Metoda percentylowa (99%):          {thresh_99:.6f}")
    print(f" -> Metoda percentylowa (90%):          {thresh_90:.6f}")
    print(f" -> Metoda percentylowa (85%):          {thresh_85:.6f}")

    thresholds = {
        "Statystyczny_3STD": thresh_std,
        "Percentyl_99": thresh_99,
        "Percentyl_90": thresh_90,
        "Percentyl_85": thresh_85
    }

    # 2. Pętla przechodząca przez każdy próg
    for name, val in thresholds.items():
        print(f"\n" + "=" * 30)
        print(f"EWALUACJA DLA PROGU: {name}")
        print(f"Wartość progu: {val:.6f}")
        print("=" * 30)

        # Klasyfikacja (0 = Benign, 1 = Atak)
        # Jeśli błąd > próg -> Atak (1), w przeciwnym razie Normalny (0)
        # Klasyfikacja binarna na podstawie aktualnego progu
        y_pred_current = (mse_errors > val).astype(int)

        # Wyświetlenie raportu w konsoli
        print(classification_report(y_test, y_pred_current, target_names=['Normalny', 'Atak']))

        # Tworzenie Macierzy Pomyłek
        cm = confusion_matrix(y_test, y_pred_current)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normalny', 'Atak'],
                    yticklabels=['Normalny', 'Atak'])
        plt.title(f'Macierz Pomyłek - {name}')
        plt.ylabel('Prawdziwa etykieta')
        plt.xlabel('Przewidziana etykieta')

        # Zapisywanie z unikalną nazwą pliku
        plt.savefig(f"confusion_matrix_{name}.png", bbox_inches='tight')
        plt.show()

        # Wykres rozkładu (opcjonalnie, jeśli Twoja funkcja to obsługuje)
        plot_error_distribution(mse_errors, y_test, val)

if __name__ == '__main__':
    main()