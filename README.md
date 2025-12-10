#  Aplikacja do Analizy Obrazu Dna Oka w Kierunku Retinopatii Cukrzycowej

**System Klasyfikacji Obrazów Dna Oka z Wykorzystaniem Głębokiego Uczenia (ConvNeXt-Tiny)**

---

## 1.  Opis Projektu

Projekt polega na stworzeniu **webowej aplikacji** do automatycznej analizy obrazów dna oka w celu wykrywania **Retinopatii Cukrzycowej (Diabetic Retinopathy)**.

Aplikacja wykorzystuje model głębokiego uczenia oparty na architekturze **ConvNeXt-Tiny**, dostosowany do zadania **klasyfikacji binarnej**:

| Klasa | Opis |
| :---: | :--- |
| **0** | Brak cech retinopatii (Zdrowe oko) |
| **1** | Obecność zmian chorobowych (Podejrzenie retinopatii) |

### Główne Funkcjonalności:
*  **Konto użytkownika:** Rejestracja i logowanie (indywidualne konto).
*  **Analiza:** Przesyłanie zdjęcia dna oka i automatyczna klasyfikacja.
*  **Wyniki:** Prezentacja wyniku wraz z prawdopodobieństwem (*score*).
*  **Historia:** Zapis historii wykonanych analiz w bazie danych.
*  **Raporty:** Możliwość pobrania raportu PDF z wynikiem dla wybranego badania.

---

## 2.  Wymagania Systemowe

* **Python:** Wersja 3.10 – 3.12
* **Menedżer pakietów:** `pip`
* **System operacyjny:** Windows lub Linux
* **Biblioteki:** Wymienione w pliku `requirements.txt`
* **Opcjonalnie:** Zainstalowany PyTorch z obsługą CUDA (jeśli planowana jest inferencja na GPU).

---

## 3. Instalacja i Uruchomienie

### Krok 1: Instalacja bibliotek
Po aktywowaniu wirtualnego środowiska zainstaluj wymagane zależności:

```bash
pip install -r requirements.txt

``` 

---

## 4. Struktura projektu

    Przykładowa struktura katalogów projektu:

    DiabeticRetinopathy/
    │
    ├── app/
    │   ├── main.py                 # Główny plik aplikacji Flask (backend)
    │   ├── models/                 # Zapisany model (.pth)
    │   │   └── best_convnext_tiny_binary.pth
    │   ├── static/                 # Pliki statyczne (CSS, obrazy, uploady)
    │   │   └── uploads/            # Przesłane przez użytkownika obrazy zapisane lokalnie
    │   ├── templates/              # Szablony HTML (frontend)
    │   ├── instance/
    │   │   └── database.db         # Lokalna baza danych SQLite
    │   ├── __init__.py             # Plik inicjujący moduł aplikacji
    │   └── key.txt                 # Klucz
    │
    │
    └── README.md                   # Dokumentacja projektu

---

## 5. Uruchomienie aplikacji lokalnie
 ### 5.1. Przejście do katalogu aplikacji

Po aktywowaniu wirtualnego środowiska przejdź do katalogu `app`:

`cd app`

### 5.2. Uruchomienie serwera Flask
Aby uruchomić aplikację w terminalu w katalogu `app`:

`python main.py`

Jeżeli wszystko przebiegnie poprawnie, aplikacja będzie dostępna pod adresem:

`http://127.0.0.1:5000/`

---


---

## 6. Korzystanie z aplikacji

Po uruchomieniu aplikacji użytkownik ma dostęp do następujących funkcji:

### 6.1. Rejestracja i logowanie

- Nowy użytkownik może założyć konto za pomocą formularza rejestracyjnego.

- Po rejestracji możliwe jest logowanie do panelu użytkownika.

- Dostęp do analizy obrazów oraz historii badań wymaga zalogowania.

### 6.2. Analiza obrazu dna oka

1. Użytkownik przechodzi do zakładki odpowiedzialnej za przesyłanie obrazu.

2. Wybiera plik ze zdjęciem dna oka (format .jpg / .jpeg / .png).

3. Backend wywołuje funkcję predykcyjną modelu (run_model_on_image), która wczytuje obraz, wykonuje preprocessing (zmiana rozmiaru), uruchamia model i wylicza prawdopodobieństwo (score).

**Interpretacja wyników**

- jeśli score ≥ 0,5 → klasyfikacja: podejrzenie retinopatii,

- jeśli score < 0,5 → klasyfikacja: brak cech retinopatii.

### 6.3. Historia badań

Każdy wynik analizy jest zapisywany w lokalnej bazie danych użytkownika (SQLite).
W historii badań prezentowane są:

- Historia badań prezentuje: miniatury obrazów, wynik klasyfikacji, wartość score oraz datę analizy.

- Aplikacja umożliwia wygenerowanie i pobranie Raportu PDF dla wybranej analizy.


### 6.4. Generowanie raportu PDF

Aplikacja umożliwia wygenerowanie raportu w formacie PDF dla wybranej analizy.
Raport zawiera m.in.:

- wynik modelu (etykieta),

- wartość prawodopodobieństwa,

- identyfikator badania oraz datę analizy.

 ---

## 7. Model klasyfikacyjny

W aplikacji wykorzystano model głębokiego uczenia oparty o architekturę ConvNeXt-Tiny: sieć została przystosowana do klasyfikacji binarnej (dwie klasy wyjściowe),
 warstwa klasyfikacyjna została zastąpiona warstwą w pełni połączoną o dwóch neuronach,
 model został wytrenowany wcześniej na zbiorze obrazów dna oka (część trenowania jest opisana w pracy dyplomowej), finalne wagi modelu zostały zapisane w pliku:
`app/models/best_convnext_tiny_binary.pth`

---

## 8. Najczęstsze problemy
### 8.1. Brak pliku modelu

Jeżeli pojawia się błąd: `FileNotFoundError: 'models/best_convnext_tiny_binary.pth'`

należy upewnić się, że plik z wagami modelu znajduje się w katalogu:

`app/models/best_convnext_tiny_binary.pth`

### 8.2. Problemy z zależnościami

Jeżeli podczas uruchamiania pojawią się błędy typu ModuleNotFoundError, należy upewnić się, że wszystkie biblioteki zostały poprawnie zainstalowane za pomocą:


`pip install -r requirements.txt`


---

## 9. Informacje końcowe

Projekt stanowi część pracy dyplomowej dotyczącej zastosowania konwolucyjnych sieci neuronowych w automatycznej analizie obrazów dna oka w kierunku wykrywania retinopatii cukrzycowej.