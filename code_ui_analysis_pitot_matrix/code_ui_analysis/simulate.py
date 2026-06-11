"""
simulate.py — generowanie syntetycznych danych dla matrycy 5 rurek Pitota.

Moduł zawiera:
  sensor_noise(V, n_samples, fs, rng) — addytywny szum czujnika [Pa], per-kanał
  (pozostałe funkcje: turbulence_series, simulate_dataset, step_alpha_profile —
   do implementacji w kolejnych etapach)

Model szumu wyekstrahowany z nagrań WindShape 2025-09-09.
Parametry wyznaczone w noise_analysis.py; pełna procedura w validate_bags.py.
"""

import numpy as np

# ── stałe fizyczne ─────────────────────────────────────────────────────────────
RHO = 1.225  # gęstość powietrza [kg/m³] (ISA, poziom morza)

# ── parametry modelu szumu (RE-WALIDACJA 2026-06-09, revalidate.py) ─────────────
#
# ⚠ MODEL PRZEBUDOWANY po re-walidacji wszystkich bagów (źródło: zawartość, nie
# nazwa). Wnioski rozstrzygające (revalidate.py):
#
# 1) SZUM CZUJNIKA = STAŁY floor ~1.05 Pa, BIAŁY (per-kanał, niezależny per ADC).
#    Potwierdzony w 7 nagraniach fan-off/low-V ORAZ przy 3 m/s (wszystkie:
#    σ≈0.94–1.12 Pa, PSD slope≈0, τ_corr≈0.1 s). NIE zależy od V.
#
# 2) Nadwyżka σ przy 6/7.5 m/s to TURBULENCJA strugi — SKORELOWANA (1/f, długie
#    τ_corr 2–2.8 s) i COMMON-MODE (wspólne wejście aero, nie tor pomiarowy).
#    σ_total na oknach stacjonarnych:  3 m/s=1.052, 6 m/s=1.876, 7.5 m/s=3.412 Pa.
#    σ_turb=√(σ_total²−σ_floor²):       3→0,  6→1.55,  7.5→3.25 Pa.
#
# ❌ Stary model σ²=σ_e²+(K·q)² (σ(7.5)=4.153) ODRZUCONY: (a) σ(7.5)=4.153 było
#    zawyżone — zawierało dryf po 220 s; prawdziwe σ na oknie stacjonarnym=3.412;
#    (b) traktowanie całego σ jako szumu per-kanał ZAWYŻA σ_α, bo turbulencja
#    (fluktuacja V) KASUJE się w znormalizowanym cp_α=(P_up−P_dn)/(P_up+P_dn).
#    Werdykt H2: σ_α maleje z V (NIE plateau). Patrz revalidate.py.
#
# UWAGA pasmowa: fs=10 Hz → floor 1.05 Pa to GÓRNE ograniczenie szumu czujnika
# (aliasing + filtr FCU). Prawdziwy σ_sensor wymaga surowego topicu ≥100 Hz.

# Szum CZUJNIKA (per-kanał, biały, niezależny od V) — to zwraca sensor_noise().
SIGMA_SENSOR = 1.05     # Pa — mediana floor z nagrań fan-off + 3 m/s
ALPHA_SENSOR = -0.1     # nachylenie PSD floor (≈ biały)

# Model TURBULENCJI strugi (COMMON-MODE, wejście aero) — do turbulence_series()
# w simulate_dataset (Etap 3). NIE jest szumem toru pomiarowego, NIE wchodzi tu.
TURB_MODEL = {
    3.0: {"sigma_dp": 0.0,  "slope": -0.4},   # brak turbulencji (floor-dominated)
    6.0: {"sigma_dp": 1.55, "slope": -1.25},  # 1/f, τ≈2.0 s
    7.5: {"sigma_dp": 3.25, "slope": -1.60},  # 1/f², τ≈2.8 s
}

# Zmierzone σ_total (floor+turbulencja) na oknach stacjonarnych — TYLKO referencja
# (np. wariant H1, górne ograniczenie). NIE używać jako szumu per-kanał dla α.
_CAL_V           = np.array([3.0,    6.0,    7.5])
_CAL_Q           = 0.5 * RHO * _CAL_V ** 2
_CAL_SIGMA_TOTAL = np.array([1.052,  1.876,  3.412])

# Kwantyzacja ADC: mediana LSB_ΔP ze wszystkich bagów
LSB_DP = 0.12  # Pa


# ── funkcje pomocnicze ─────────────────────────────────────────────────────────

def _q_from_v(V: float) -> float:
    """Ciśnienie dynamiczne [Pa]: q = 0.5·ρ·V²  (V ≥ 0)."""
    return 0.5 * RHO * float(V) ** 2


def sigma_sensor(V=None) -> float:
    """
    σ szumu CZUJNIKA [Pa] — stały floor, niezależny od V (werdykt H2).

    Parametr V akceptowany dla zgodności interfejsu, ale ignorowany: re-walidacja
    pokazała brak zależności σ_sensor od prędkości (cała zależność od V to
    turbulencja common-mode, modelowana osobno w TURB_MODEL).
    """
    return SIGMA_SENSOR


def sigma_total_measured(V: float) -> float:
    """
    Zmierzone σ_total (floor+turbulencja) na oknie stacjonarnym [Pa] — REFERENCJA.

    Do porównań / wariantu H1 (górne ograniczenie). NIE jest to szum per-kanał
    istotny dla kanału α — turbulencja kasuje się w cp_α. Patrz nagłówek modułu.
    """
    return float(np.interp(_q_from_v(V), _CAL_Q, _CAL_SIGMA_TOTAL))


# Alias zachowany dla zgodności wstecznej (kod, który importował _sigma_from_q).
# ⚠ Zwraca teraz STAŁY floor czujnika (H2), nie stary model σ∝q.
def _sigma_from_q(q: float) -> float:
    """σ szumu czujnika [Pa] — stały floor (H2). q ignorowane (patrz sigma_sensor)."""
    return SIGMA_SENSOR


def _alpha_from_q(q: float) -> float:
    """Nachylenie widmowe szumu czujnika — ≈ biały, niezależne od q (H2)."""
    return ALPHA_SENSOR


def _colored_noise_fft(n: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """
    Generuje szum barwny długości n o widmie gęstości mocy ∝ f^alpha metodą FFT.

    alpha =  0.0 → szum biały
    alpha = -1.0 → szum różowy (1/f)
    alpha = -2.0 → szum czerwony / Browna (1/f²)

    Zwraca sygnał o zerowej średniej i jednostkowej wariancji.
    """
    f = np.fft.rfftfreq(n)   # [0, 1/n, 2/n, ..., 0.5]
    f[0] = 1.0               # tymczasowo — zerujemy DC po ukształtowaniu

    # Amplituda ~ f^(α/2), żeby PSD ~ f^α
    amplitude = f ** (alpha / 2.0)
    amplitude[0] = 0.0       # brak składowej stałej

    # Losowe fazy
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(f))
    spectrum = amplitude * np.exp(1j * phases)

    # Bin Nyquista musi być rzeczywisty (dla n parzystego)
    if n % 2 == 0:
        spectrum[-1] = spectrum[-1].real

    noise = np.fft.irfft(spectrum, n=n)

    # Normalizacja do σ=1
    std = noise.std()
    if std > 0.0:
        noise /= std
    return noise


# ── główna funkcja ─────────────────────────────────────────────────────────────

def sensor_noise(
    V: float,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Addytywny szum pojedynczego kanału CZUJNIKA Pitota [Pa] (tor pomiarowy).

    Szum jest addytywny w domenie ciśnienia różnicowego.  Dodaj do
    niezaszumionego ΔP = sign(V) · 0.5 · ρ · V²  każdego kanału osobno,
    z NIEZALEŻNYM ziarnem rng per sonda (każdy ADC ma własny szum).

    ⚠ Po re-walidacji (revalidate.py): to jest WYŁĄCZNIE szum czujnika —
    STAŁY floor ~1.05 Pa, biały, NIEZALEŻNY od V i niezależny per kanał.
    Turbulencja strugi (skorelowana, common-mode, rosnąca z V) NIE wchodzi
    tutaj — modelowana osobno (TURB_MODEL) jako wspólne wejście aerodynamiczne
    w simulate_dataset, bo w znormalizowanym cp_α=(P_up−P_dn)/(P_up+P_dn)
    fluktuacja prędkości się kasuje.

    Parametry
    ----------
    V : float
        Prędkość napływu [m/s].  Akceptowana dla zgodności interfejsu, ale
        σ_sensor NIE zależy od V (werdykt H2). Użyj rzeczywistej prędkości
        tunelu, nie odczytu FCU (FCU zaniża — patrz CLAUDE.md).
    n_samples : int
        Liczba próbek do wygenerowania.
    fs : float
        Częstotliwość próbkowania [Hz].  Akceptowany dla zgodności interfejsu.
    rng : np.random.Generator
        Generator liczb losowych, np. np.random.default_rng(42).

    Zwraca
    -------
    noise : np.ndarray, kształt (n_samples,)
        Szum addytywny czujnika [Pa].

    Model
    -----
    1. σ_sensor = SIGMA_SENSOR = 1.05 Pa, STAŁE (niezależne od V).
    2. Barwa widmowa ≈ biała (ALPHA_SENSOR = -0.1).
    3. Kwantyzacja ADC: zaokrąglenie do LSB_DP = 0.12 Pa.

    Ograniczenia
    ------------
    - fs = 10 Hz → Nyquist = 5 Hz.  Biały szum elektroniczny (>20 Hz) jest
      aliasowany; floor 1.05 Pa to GÓRNE ograniczenie szumu czujnika.
    - Rozkład nie-gaussowski (kwantyzacja) — nie modelowany explicite;
      generowany szum jest gaussowski po sformowaniu widma + kwantyzacja.
    - Outliery (>5σ) mają empiryczną częstość < 0.03% — pominięte.

    Przykład
    --------
    >>> rng = np.random.default_rng(0)
    >>> noise = sensor_noise(V=6.0, n_samples=1000, fs=10.0, rng=rng)
    >>> abs(noise.std() - 1.05) < 0.1  # stały floor, niezależny od V
    True
    """
    q     = _q_from_v(V)
    sigma = sigma_sensor(V)
    alpha = _alpha_from_q(q)

    # Szum barwny o jednostkowej wariancji
    noise = _colored_noise_fft(n_samples, alpha, rng)

    # Skalowanie do σ_ΔP(V) i kwantyzacja
    noise = noise * sigma
    noise = np.round(noise / LSB_DP) * LSB_DP

    return noise.astype(np.float32)


# ── profil schodkowy kąta natarcia (robot Stäubli) ─────────────────────────────

def step_alpha_profile(fs, levels_deg, dwell_s):
    """
    Schodkowy profil kąta natarcia α(t) [deg] — odwzorowuje ruch robota Stäubli.

    Robot ustawia model na kolejnych zadanych kątach (levels_deg), utrzymując każdy
    przez dwell_s sekund. Krawędzie są idealnie ostre (pozycja przegubów = GT).

    Parametry
    ----------
    fs : float           — częstotliwość próbkowania [Hz]
    levels_deg : iterable — kolejne poziomy α [deg], np. [-25,-15,...,25]
    dwell_s : float      — czas utrzymania każdego poziomu [s]

    Zwraca: np.ndarray α(t) [deg], długość len(levels)·round(dwell_s·fs).
    """
    n_dwell = int(round(dwell_s * fs))
    return np.concatenate([np.full(n_dwell, float(lvl)) for lvl in levels_deg])


# ── szybki test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal as sp_signal

    rng = np.random.default_rng(42)
    fs  = 10.0
    N   = 4000   # 400 s @ 10 Hz

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.ravel()

    cases = [
        (0.0,   "V=0 m/s (fan off)"),
        (3.0,   "V=3 m/s setpoint"),
        (6.0,   "V=6 m/s setpoint"),
        (7.5,   "V=7.5 m/s setpoint"),
        (10.0,  "V=10 m/s (ekstrapolacja)"),
    ]
    # szum CZUJNIKA = stały floor (H2), niezależny od V

    for ax, (V, title) in zip(axes, cases):
        noise = sensor_noise(V, N, fs, rng)
        q     = _q_from_v(V)
        sigma_gen  = noise.std()
        sigma_exp  = sigma_sensor(V)
        alpha_used = _alpha_from_q(q)

        # PSD
        f_psd, pxx = sp_signal.welch(noise, fs=fs,
                                     nperseg=min(N // 4, 256),
                                     scaling="density")
        f_psd, pxx = f_psd[f_psd > 0], pxx[f_psd > 0]

        ax.loglog(f_psd, pxx, lw=1.2, label=f"generated  σ={sigma_gen:.3f} Pa")
        ax.axhline(sigma_exp ** 2 / (fs / 2), color="r", ls="--", lw=1,
                   label=f"expected σ={sigma_exp:.3f} Pa (flat)")
        ax.set_title(f"{title}\nα={alpha_used:.2f}  σ_gen={sigma_gen:.3f} Pa")
        ax.set_xlabel("Częstotliwość [Hz]")
        ax.set_ylabel("PSD [(Pa)²/Hz]")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    # Ukryj puste panele
    for ax in axes[len(cases):]:
        ax.set_visible(False)

    plt.suptitle("sensor_noise() — weryfikacja PSD dla różnych prędkości", fontsize=12)
    plt.tight_layout()
    plt.savefig("sensor_noise_test.png", dpi=130)
    print("Zapisano sensor_noise_test.png")

    print("\nTabela weryfikacyjna (szum czujnika = stały floor, H2):")
    print(f"  {'V [m/s]':>8}  {'q [Pa]':>8}  {'α':>6}  {'σ_sensor':>9}  {'σ_gen':>9}  {'błąd%':>7}  {'σ_total*':>9}")
    print("  " + "-" * 72)
    rng2 = np.random.default_rng(0)
    for V, _ in cases:
        q     = _q_from_v(V)
        sigma_m = sigma_sensor(V)
        alpha   = _alpha_from_q(q)
        noise   = sensor_noise(V, 20000, fs, rng2)
        sigma_g = noise.std()
        err     = (sigma_g - sigma_m) / sigma_m * 100
        s_tot   = sigma_total_measured(V) if 3.0 <= V <= 7.5 else float("nan")
        print(f"  {V:>8.1f}  {q:>8.3f}  {alpha:>6.2f}  {sigma_m:>9.4f}  {sigma_g:>9.4f}  "
              f"{err:>+7.2f}%  {s_tot:>9.3f}")
    print("  * σ_total = floor+turbulencja zmierzone (referencja); NIE szum per-kanał dla α.")
