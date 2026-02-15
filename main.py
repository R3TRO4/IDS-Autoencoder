from pathlib import Path
from preprocessing import *
from train_model import *
from evaluate_model import *

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
    # X_train, X_val = load_train_data()
    #
    # if X_train is None or X_val is None:
    #     return
    #
    # # Pobieramy liczbę cech (kolumn) dynamicznie z danych
    # input_dim = X_train.shape[1]
    #
    # # Budowa modelu
    # model = build_autoencoder(input_dim)
    # model.summary() # Wyświetla tabelkę z architekturą w konsoli
    #
    # # Konfiguracja Callbacków (mechanizmów kontrolnych)
    # callbacks = [
    #     # EarlyStopping: Jeśli val_loss nie spadnie przez 5 epok, przewij uczenie.
    #     # Ten machanizm zapobiega przeuczeniu (overfitting)
    #
    #     # ModelCheckpoint: Zapisuj model tylko wyedy, gdy jest najlepszy (najmniejszy błąd)
    #     ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=0)
    # ]
    #
    # # Trenowanie modelu
    # print("\n[INFO] Rozpoczęcie treningu sieci...")
    # # UWAGA: W Autoencoderze X (wejście) jest równe Y (oczekiwane wyjście)!
    # # Dkategi oidaheny x=X_train, y=X_train
    # history = model.fit(
    #     x=X_train,
    #     y=X_train,
    #     epochs=50,          # Maksymalna liczba epok (EarlyStopping i tak przerwie wcześniej)
    #     batch_size=256,     # Ile próbek na raz mieli karta graficzna/CPU
    #     shuffle=True,       # Mieszanie danych w każdej epoce
    #     validation_data=(X_val, X_val), # Sprawdzanie jakości na zbiorze walidacyjnym
    #     callbacks=callbacks
    # )
    #
    # # Zapisz finalny model i wykres
    # print("[INFO] Trening zakończony.")
    # plot_history(history)
    #
    # # Model jest już zapisany przez Checkpoint jako 'best_model.keras'
    # print("Najlepszy model zapisano jako: best_model.keras")


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
    thresh_std, thresh_99 = find_threshold_statistics(mse_errors, y_test)

    print(f"\nSugerowane progi:")
    print(f" -> Metoda statystyczna (Mean + 3*Std): {thresh_std:.6f}")
    print(f" -> Metoda percentylowa (99%):          {thresh_99:.6f}")

    # WYBÓR PROGU DO OCENY (Możesz tu zmienić na thresh_std lub wpisać własną liczbę)
    # FINAL_THRESHOLD = thresh_std
    FINAL_THRESHOLD = thresh_99
    print(f"\n[INFO] Przyjęty próg do klasyfikacji: {FINAL_THRESHOLD:.6f}")

    # Klasyfikacja (0 = Benign, 1 = Atak)
    # Jeśli błąd > próg -> Atak (1), w przeciwnym razie Normalny (0)
    y_pred = (mse_errors > FINAL_THRESHOLD).astype(int)

    # Raport wyników
    print("\n=== RAPORT KLASYFIKACJI ===")
    print(classification_report(y_test, y_pred, target_names=['Normalny', 'Atak']))

    # Macierz pomyłek
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normalny', 'Atak'],
                yticklabels=['Normalny', 'Atak'])
    plt.title('Macierz Pomyłek (Confusion Matrix)')
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Przewidziana etykieta')
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Wykres rozkładu błędów
    plot_error_distribution(mse_errors, y_test, FINAL_THRESHOLD)

if __name__ == '__main__':
    main()