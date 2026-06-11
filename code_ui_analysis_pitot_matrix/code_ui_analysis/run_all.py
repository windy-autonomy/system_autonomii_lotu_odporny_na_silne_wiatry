"""
run_all.py — potok end-to-end matrycy 5 rurek Pitota (dane SYMULOWANE).

Kolejność:
  1. (opcjonalnie) re-walidacja bagów WindShape — tylko gdy podasz --revalidate
     (wymaga katalogu windwall/ z nagraniami).
  2. Macierz eksperymentów degradacyjnych (experiments.py) → results/experiments.json
  3. Wykresy pod raport (plotting.py) → SKNTI/img/*.png

Sanity fizyki (roundtrip forward→estymator) jest w physics.py i estimator.py
(uruchom je osobno: `python3 physics.py`, `python3 estimator.py`).

Użycie:
  python3 run_all.py                # eksperymenty + wykresy
  python3 run_all.py --revalidate   # dodatkowo re-walidacja bagów (Etap 0)
"""

import sys
import experiments
import plotting


def main():
    if "--revalidate" in sys.argv:
        print(">>> Etap 0: re-walidacja bagów WindShape\n")
        import revalidate
        revalidate.main()
        print()

    print(">>> Etap eksperymentów (Monte-Carlo degradacja)\n")
    experiments.main()
    print()

    print(">>> Generowanie wykresów do SKNTI/img/\n")
    plotting.main()
    print("\n>>> Potok zakończony. Wyniki: results/experiments.json, SKNTI/img/*.png")


if __name__ == "__main__":
    main()
