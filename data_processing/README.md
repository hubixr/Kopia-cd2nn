# Przetwarzanie pomiarów

Dane źródłowe są w `inputs/`, a wygenerowane obrazy w `outputs/`.

```text
data_processing/
├── inputs/
│   ├── doe/
│   ├── soczewka/
│   ├── pierwsza_warstwa/
│   ├── druga_warstwa/
│   ├── obie_warstwy/
│   ├── porownanie_mocy/
│   └── wzorcowa/
├── outputs/
│   ├── singlefreq/      # przekroje Y–Z; opcjonalnie wszystkie przekroje X–Y
│   ├── zscan/           # przekroje X–Y dla wybranych pozycji Z
│   ├── multifreq/       # przekroje i zestawienia częstotliwości
│   └── archive/         # obrazy zachowane sprzed uporządkowania
├── main_singlefreq.py
├── main_singlefreq_zscan.py
└── main_multifreq.py
```

Foldery `zscan/` i `multifreq/` powstają po uruchomieniu odpowiednich skryptów.

## Dodawanie pomiarów

Umieść plik `.npy` w `inputs/<seria>/`. Towarzyszące pliki `.meta` i `.txt`
trzymaj obok niego. Plik metadanych powinien mieć tę samą nazwę podstawową,
np. `pomiar.npy` i `pomiar.meta`. Możesz dodać nowy folder serii.

Obsługiwany jest także wcześniejszy układ
`inputs/<seria>/dane z pomiarów/<podseria>/*.npy`.
Wyniki zachowują strukturę podfolderów wejściowych.

## Uruchamianie

W tym folderze:

```bash
python3 main_singlefreq.py
python3 main_singlefreq_zscan.py
python3 main_multifreq.py
```

W środowisku bez okien graficznych dodaj `MPLBACKEND=Agg` przed poleceniem.
Ścieżki wejścia i wyjścia są ustalane względem skryptu, więc można go też
uruchomić przez pełną ścieżkę z innego katalogu.

Każdy pomiar ma własny folder, np.
`outputs/singlefreq/obie_warstwy/obie_warstwy_195-205cm/`.
Ponowne uruchomienie nadpisuje obrazy tego samego pomiaru w danym trybie.

[Spis wyników singlefreq](outputs/singlefreq/README.md)

## Ustawienia i metadane

Ustawienia znajdują się na początku skryptów. `GENERATE_PROPAGATION_SCANS`
włącza dodatkowe obrazy X–Y. W `main_singlefreq_zscan.py` pozycje Z wybiera się
przez `Z_TARGETS_MM` albo `Z_RANGE_MM`.

`main_singlefreq.py` przetwarza wszystkie znalezione `.npy`:
`TARGET_FREQ_GHZ` jest obecnie tylko informacją w komunikacie, bez filtrowania.

`inputs/doe/DOEv0.npy` nie ma odpowiadającego pliku `DOEv0.meta`.
Skrypty singlefreq i multifreq używają dla niego parametrów domyślnych;
zscan ma istniejący mechanizm wyboru innego `.meta` z tego samego folderu.
Sprawdź właściwe metadane przed interpretacją pozycji Z tego pomiaru.
