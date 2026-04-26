# Chaotyczne przekształcanie obrazu cyfrowego

Projekt przedstawia aplikację do chaotycznego przekształcania obrazu cyfrowego zrealizowaną w języku Python

Program umożliwia wykonanie trzech etapów przekształcania obrazu

Etap 1 pokazuje naiwny scrambling obrazu

Etap 2 pokazuje czystą permutację pikseli sterowaną kluczem

Etap 3 rozszerza permutację o mechanizm wzmacniający w postaci substytucji XOR

Projekt ma charakter edukacyjny i pokazuje że wizualny chaos obrazu nie oznacza bezpieczeństwa danych

## Funkcjonalności

wczytanie obrazu PNG JPEG BMP

wybór etapu 1 2 3

wprowadzanie poprawnego i błędnego klucza

scrambling obrazu

unscrambling obrazu

porównanie wyniku dla poprawnego i błędnego klucza

wyświetlanie obrazu oryginalnego przekształconego i odtworzonego

obliczanie korelacji sąsiednich pikseli

obliczanie różnicy obrazu po błędnym kluczu

zapis wyników do plików

## Technologie

Python

NumPy

Pillow

Tkinter

## Uruchomienie projektu

Pobierz lub sklonuj repozytorium

Zainstaluj wymagane biblioteki

```bash
pip install numpy pillow
