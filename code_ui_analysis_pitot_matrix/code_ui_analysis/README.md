# code_ui_analysis

Kod aplikacji UI oraz potoku analizy danych dla stanowiska matrycy **5 rurek
Pitota** (tunel WindShape + robot Stäubli + czujnik siły 6-osiowy). Folder
zawiera **wyłącznie kod źródłowy** — bez nagrań bagów ROS2, obrazów i archiwów.

Dane wejściowe są w pełni **symulowane** (zgodnie z zasadą *inverse crime*:
generator danych jest celowo bogatszy niż estymator). Parametry szumu zostały
wyekstrahowane z rzeczywistych nagrań tunelu WindShape z 2025-09-09.

## Instalacja

```bash
pip install -r requirements.txt
```

Wymaga Pythona 3.10+. Zależności: numpy, scipy, matplotlib, plotly, dash,
dash-bootstrap-components.

## Aplikacja UI

```bash
python3 ui_app.py
# otwórz http://127.0.0.1:8050
```

`ui_app.py` — interfejs Dash (prototyp/demo). Zakładki:

| Zakładka | Funkcja |
|---|---|
| **Monitor** | replay baga ROS2: airspeed + IMU pitch |
| **Trajektoria** | projektant profilu schodkowego i sweepowego |
| **Rejestracja** | zarządzanie nagraniami (ROS2 bag) |
| **Analiza** | krzywe RMSE z `results/experiments.json` |

> ⚠ **Dane dla UI nie są w tym folderze.**
> - Zakładka *Monitor* czyta bag ze ścieżki na sztywno w `ui_app.py` (`_BAG_PATH`,
>   domyślnie `windwall/6ms_.../...db3`). Aby działała, dorzuć nagranie lub popraw
>   tę ścieżkę.
> - Zakładka *Analiza* czyta dołączony `results/experiments.json` (wygenerowany
>   przez `experiments.py`).

## Potok analizy danych

End-to-end (eksperymenty Monte-Carlo + wykresy):

```bash
python3 run_all.py                # eksperymenty + wykresy
python3 run_all.py --revalidate   # dodatkowo re-walidacja bagów (wymaga windwall/)
```

### Moduły pipeline

| Plik | Rola |
|---|---|
| `physics.py` | model fizyczny — **generator** „prawdy" (cos^n, bias montażowy, crosstalk) |
| `simulate.py` | generowanie danych syntetycznych + model szumu czujnika [Pa] |
| `estimator.py` | **estymator A** (różnice par) + filtr Kalmana 1D [V, α, β] |
| `experiments.py` | macierz eksperymentów degradacyjnych → `results/experiments.json` |
| `plotting.py` | wykresy pod raport SKNTI → `SKNTI/img/*.png` |
| `run_all.py` | spina cały potok |

### Skrypty analizy nagrań (wymagają katalogu `windwall/`)

| Plik | Rola |
|---|---|
| `noise_analysis.py` | charakteryzacja szumu czujnika z nagrań tunelu (PSD, histogram, ACF, fit) |
| `validate_bags.py` | protokół walidacji bagów (timing, declared-vs-measured, flagi, werdykt) |
| `revalidate.py` | pełna re-walidacja od nowa — źródłem prawdy jest tylko zawartość bagów |

Sanity fizyki (roundtrip generator → estymator) uruchom osobno:

```bash
python3 physics.py
python3 estimator.py
```

## Estymator A (ustalony)

```
V      = sqrt(2·ΔP_C / ρ)
β      = arcsin((P_R − P_L) / ((P_R + P_L) · k_β))
α      = arcsin((P_up − P_dn) / ((P_up + P_dn) · k_α))
V_corr = V / cos(sqrt(α² + β²))
```

Potok: filtr medianowy outlierów → estymator A → Kalman 1D per kanał.

## Struktura folderu

```
code_ui_analysis/
├── README.md
├── requirements.txt
├── ui_app.py              # aplikacja UI (Dash)
├── run_all.py             # potok end-to-end
├── physics.py             # generator danych
├── simulate.py           # szum + symulacja
├── estimator.py           # estymator A + Kalman
├── experiments.py         # eksperymenty degradacyjne
├── plotting.py            # wykresy
├── noise_analysis.py      # analiza szumu z bagów
├── validate_bags.py       # walidacja bagów
├── revalidate.py          # re-walidacja
└── results/
    └── experiments.json   # wynik eksperymentów (czytany przez UI)
```
