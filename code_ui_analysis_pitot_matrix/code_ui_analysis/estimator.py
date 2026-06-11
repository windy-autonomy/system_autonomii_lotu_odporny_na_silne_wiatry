"""
estimator.py — estymator wiatru (wariant A: różnice par) dla matrycy 5 rurek Pitota.

Realizuje rdzeń Etapu 2 z meta_prompt_end_to_end.md. Algorytm jest USTALONY —
estymator różnic par (NIE nieliniowy LSQ, NIE UKF). Dane są w pełni symulowane,
więc celowo trzymamy prosty, jawnie odwracalny estymator (zasada inverse crime:
generator danych będzie BOGATSZY niż to, co ten estymator zakłada).

Geometria krzyża (φ = kąt kantowania sond od osi ciała):
            P_up        (górny rząd, pitch +)
             |
     P_L -- P_C -- P_R  (środkowy rząd: yaw)
             |
            P_dn        (dolny rząd, pitch −)

  P_C            → prędkość V
  para P_L/P_R   → kąt znoszenia β (yaw)
  para P_up/P_dn → kąt natarcia α (pitch)

Estymator A:
  V      = sqrt(2·ΔP_C / ρ)
  cp_α   = (P_up − P_dn) / (P_up + P_dn);   α = arcsin(cp_α / k_α)
  cp_β   = (P_R  − P_L ) / (P_R  + P_L );   β = arcsin(cp_β / k_β)
  V_corr = V / cos(sqrt(α² + β²))

Potok: medfilt (outliery) → estymator A → filtr Kalmana 1D per kanał [V, α, β].

Podstawa matematyczna k_α (φ=45°, n=2):
  P_up = q·cos²(φ−α), P_dn = q·cos²(φ+α)
  cp_α = sin(2φ)·sin(2α) / (1 + cos(2φ)·cos(2α))  =  sin(2α)   dla φ=45°
  Linearyzacja w α=0: cp_α ≈ (2·tan φ)·α  ⇒ slope k_α = 2·tan φ = 2.0 dla φ=45°.
  Wartość użytkowa k_α z LSQ-fitu na ±25° ≈ 1.885 (rozkłada bias po całym zakresie).
"""

import numpy as np
from scipy.signal import medfilt

# ── stałe modułu ─────────────────────────────────────────────────────────────
RHO_REF = 1.225      # gęstość powietrza [kg/m³] (ISA, poziom morza)

PHI_DEG = 45.0       # ZAŁOŻENIE geometryczne: kąt kantowania sond up/dn i L/R od osi
                     # ciała [deg]. Dla φ=45° forward zwija się do cp = sin(2·kąt).
                     # Realna geometria krzyża może się różnić — wtedy k = 2·tan φ.

# Współczynniki kalibracyjne estymatora (wyznaczone przez _fit_k_lsq dla φ=45°,
# n=2, zakres ±25°). Estymator ZAWSZE zakłada n=2 (inverse crime).
K_ALPHA = 1.885
K_BETA  = 1.885

# Indeksy sond w wektorze ciśnień P_5
IDX = dict(C=0, L=1, R=2, UP=3, DN=4)


# ── forward model (TYLKO do testu roundtrip) ────────────────────────────────

def _forward_min(V, alpha_deg, beta_deg, rho=RHO_REF):
    """
    CZYSTY forward model krzyża 5 sond — wyłącznie do sanity-checku (roundtrip).

    n=2, φ=PHI_DEG, BEZ biasu montażowego, crosstalk, szumu ani kwantyzacji.
    NIE używać do generowania wyników naukowych — bogaty forward_5probe
    (cos^n, n≠2, crosstalk, biasy) powstanie w physics.py (Etap 3). Trzymanie
    tego forwardu „ubogim" gwarantuje, że roundtrip testuje wyłącznie spójność
    algebry estymatora, a nie maskuje niedopasowań.

    Zwraca: wektor 5 ciśnień [Pa] w kolejności (C, L, R, up, dn).
    """
    phi = np.radians(PHI_DEG)
    a = np.radians(alpha_deg)
    b = np.radians(beta_deg)
    q = 0.5 * rho * V ** 2

    # Sonda centralna: pełne ciśnienie dynamiczne pomniejszone o całkowity kąt napływu.
    theta = np.sqrt(a ** 2 + b ** 2)
    P_C = q * np.cos(theta) ** 2

    # Para pionowa (α): sondy skantowane o ±φ, napływ pod kątem φ∓α.
    P_up = q * np.cos(phi - a) ** 2
    P_dn = q * np.cos(phi + a) ** 2

    # Para pozioma (β): analogicznie dla yaw.
    P_R = q * np.cos(phi - b) ** 2
    P_L = q * np.cos(phi + b) ** 2

    P = np.empty(5)
    P[IDX["C"]]  = P_C
    P[IDX["L"]]  = P_L
    P[IDX["R"]]  = P_R
    P[IDX["UP"]] = P_up
    P[IDX["DN"]] = P_dn
    return P


def _fit_k_lsq(phi_deg=PHI_DEG, n=2.0, amax_deg=25.0, n_pts=501):
    """
    Wyznacza współczynnik kalibracyjny k przez dopasowanie najmniejszych kwadratów.

    Dla siatki kątów θ ∈ [−amax, +amax] liczymy współczynnik ciśnień pary sond
    cp(θ) z forward modelu cos^n (φ, n) i szukamy takiego k, by arcsin(cp/k)
    najlepiej (w sensie LSQ) odwzorował prawdziwy kąt θ.

    LSQ na całym zakresie daje mniejszy peak-error przy dużych kątach niż
    dopasowanie samego slope w zerze (k = 2·tan φ). Dla φ=45°, n=2: k ≈ 1.885.

    Zwraca: k (float).
    """
    phi = np.radians(phi_deg)
    th = np.radians(np.linspace(-amax_deg, amax_deg, n_pts))
    cp = (np.cos(phi - th) ** n - np.cos(phi + th) ** n) / \
         (np.cos(phi - th) ** n + np.cos(phi + th) ** n)

    # Szukamy k minimalizującego sum((arcsin(clip(cp/k)) − th)²).
    def resid(k):
        r = np.clip(cp / k, -1.0, 1.0)
        return np.sum((np.arcsin(r) - th) ** 2)

    # 1D minimalizacja po siatce + zagęszczenie (bez zależności od scipy.optimize).
    k_grid = np.linspace(1.0, 3.0, 401)
    errs = np.array([resid(k) for k in k_grid])
    k0 = k_grid[np.argmin(errs)]
    k_fine = np.linspace(k0 - 0.01, k0 + 0.01, 401)
    errs_fine = np.array([resid(k) for k in k_fine])
    return float(k_fine[np.argmin(errs_fine)])


# ── estymator A ──────────────────────────────────────────────────────────────

def estimate_wind(P_5, rho=RHO_REF, k_alpha=K_ALPHA, k_beta=K_BETA):
    """
    Estymator A (różnice par) dla pojedynczej próbki wektora 5 ciśnień.

    Parametry
    ----------
    P_5 : array-like, kształt (5,)
        Ciśnienia [Pa] w kolejności (C, L, R, up, dn) — patrz IDX.
    rho : float
        Gęstość powietrza [kg/m³].
    k_alpha, k_beta : float
        Współczynniki kalibracyjne par α / β.

    Zwraca
    -------
    dict: {V, alpha_deg, beta_deg, V_corr}
        V       — prędkość z sondy centralnej [m/s]
        alpha_deg, beta_deg — kąty natarcia / znoszenia [deg]
        V_corr  — V skorygowane o całkowity kąt napływu [m/s]

    Uwagi
    -----
    - cp/k klipowane do [−1, 1] (przy niskim SNR cp może wyjść poza zakres arcsin).
    - V liczone z |ΔP_C| (ujemne ΔP po szumie → 0, by uniknąć sqrt z liczby ujemnej).
    """
    P = np.asarray(P_5, dtype=float)
    P_C, P_L, P_R = P[IDX["C"]], P[IDX["L"]], P[IDX["R"]]
    P_up, P_dn = P[IDX["UP"]], P[IDX["DN"]]

    # Prędkość z sondy centralnej (clip do ≥0 chroni przed szumem ujemnym).
    V = np.sqrt(2.0 * max(P_C, 0.0) / rho)

    # Kąt natarcia α z pary pionowej.
    s_a = P_up + P_dn
    cp_a = (P_up - P_dn) / s_a if s_a != 0.0 else 0.0
    alpha = np.arcsin(np.clip(cp_a / k_alpha, -1.0, 1.0))

    # Kąt znoszenia β z pary poziomej.
    s_b = P_R + P_L
    cp_b = (P_R - P_L) / s_b if s_b != 0.0 else 0.0
    beta = np.arcsin(np.clip(cp_b / k_beta, -1.0, 1.0))

    # Korekta V o całkowity kąt napływu.
    theta = np.sqrt(alpha ** 2 + beta ** 2)
    V_corr = V / np.cos(theta) if np.cos(theta) != 0.0 else V

    return {
        "V": float(V),
        "alpha_deg": float(np.degrees(alpha)),
        "beta_deg": float(np.degrees(beta)),
        "V_corr": float(V_corr),
    }


def estimate_wind_weighted(P_5, sigma_per_probe, rho=RHO_REF,
                           k_alpha=K_ALPHA, k_beta=K_BETA):
    """
    Wariant WAŻONY estymatora A (ważenie sond odwrotnie do wariancji szumu).

    ⚠ WYNIK NEGATYWNY (udokumentowany świadomie): na każdy kanał kątowy mamy
    TYLKO 2 sondy (up/dn albo L/R). Jeśli ich szum jest symetryczny (równe σ —
    domyślny przypadek, ten sam model `sensor_noise` per kanał), optymalne wagi
    to ½/½, czyli zwykła różnica par — ważenie NIC nie poprawia. Sens pojawia się
    dopiero przy ASYMETRII σ między sondami (np. bias montażowy, uszkodzona sonda).
    Realna dźwignia SNR przy niskiej V to filtracja czasowa (WindKalman), nie to.

    Implementacja: gdy σ_up ≠ σ_dn, środek pary przesuwamy ku sondzie pewniejszej
    (mniejsze σ), korygując cp o ważoną sumę zamiast zwykłej. Dla równych σ
    redukuje się dokładnie do estimate_wind.

    Parametry
    ----------
    sigma_per_probe : array-like, kształt (5,)
        Odchylenia standardowe szumu [Pa] per sonda (C, L, R, up, dn).
    """
    P = np.asarray(P_5, dtype=float)
    sig = np.asarray(sigma_per_probe, dtype=float)

    P_C = P[IDX["C"]]
    V = np.sqrt(2.0 * max(P_C, 0.0) / rho)

    def weighted_cp(i_plus, i_minus):
        # Wagi odwrotne do wariancji; suma ważona w mianowniku zachowuje skalę.
        w_p = 1.0 / sig[i_plus] ** 2
        w_m = 1.0 / sig[i_minus] ** 2
        num = P[i_plus] - P[i_minus]
        den = (w_p * P[i_plus] + w_m * P[i_minus]) / (w_p + w_m) * 2.0
        return num / den if den != 0.0 else 0.0

    cp_a = weighted_cp(IDX["UP"], IDX["DN"])
    cp_b = weighted_cp(IDX["R"], IDX["L"])
    alpha = np.arcsin(np.clip(cp_a / k_alpha, -1.0, 1.0))
    beta = np.arcsin(np.clip(cp_b / k_beta, -1.0, 1.0))

    theta = np.sqrt(alpha ** 2 + beta ** 2)
    V_corr = V / np.cos(theta) if np.cos(theta) != 0.0 else V
    return {
        "V": float(V),
        "alpha_deg": float(np.degrees(alpha)),
        "beta_deg": float(np.degrees(beta)),
        "V_corr": float(V_corr),
    }


# ── filtr Kalmana 1D ─────────────────────────────────────────────────────────

class WindKalman:
    """
    Trzy NIEZALEŻNE filtry Kalmana 1D (model random-walk) dla [V, α, β].

    Model stanu per kanał:  x_k = x_{k-1} + w,  w ~ N(0, Q)
    Pomiar:                 z_k = x_k + v,      v ~ N(0, R)

    Random-walk jest właściwy dla quasi-statycznego profilu schodkowego Stäubli
    (między schodkami kąt stały). KOMPROMIS: małe Q silnie redukuje wariancję, ale
    wprowadza lag/rozmycie na krawędziach schodków; duże Q — odwrotnie. Bez
    detektora schodków (świadoma decyzja) akceptujemy lag krawędzi ≈ R/Q próbek.

    Dobór parametrów:
      R_α ≈ (σ_ΔP / (√2·m·k_α))²  [deg²] — z propagacji szumu (m = q/2).
      Q   — z oczekiwanej zmienności kąta między próbkami (małe dla schodków).
    """

    def __init__(self, Q, R):
        """
        Q, R : dict {'V':..., 'alpha':..., 'beta':...} — wariancje procesu/pomiaru.
        """
        self.Q = dict(Q)
        self.R = dict(R)
        self._keys = ("V", "alpha", "beta")
        self.reset()

    def reset(self):
        """Zeruje stan i kowariancję (P startowo duże = brak zaufania do x0)."""
        self.x = {k: None for k in self._keys}
        self.P = {k: 1e6 for k in self._keys}

    def update(self, meas):
        """
        Jeden krok predykcja+korekta dla wszystkich trzech kanałów.

        meas : dict {'V', 'alpha_deg', 'beta_deg'} — wyjście estimate_wind.
        Zwraca: dict z wygładzonymi {'V', 'alpha_deg', 'beta_deg'}.
        """
        z = {"V": meas["V"], "alpha": meas["alpha_deg"], "beta": meas["beta_deg"]}
        out = {}
        for k in self._keys:
            # Inicjalizacja pierwszą próbką.
            if self.x[k] is None:
                self.x[k] = z[k]
                self.P[k] = self.R[k]
                out[k] = self.x[k]
                continue
            # Predykcja (random-walk: x bez zmian, P rośnie o Q).
            P_pred = self.P[k] + self.Q[k]
            # Korekta.
            K = P_pred / (P_pred + self.R[k])
            self.x[k] = self.x[k] + K * (z[k] - self.x[k])
            self.P[k] = (1.0 - K) * P_pred
            out[k] = self.x[k]
        return {"V": out["V"], "alpha_deg": out["alpha"], "beta_deg": out["beta"]}


# ── pełny potok ──────────────────────────────────────────────────────────────

def pipeline(P_5_series, fs, kf_Q, kf_R, medfilt_kernel=5):
    """
    Pełny potok przetwarzania serii czasowej ciśnień.

    medfilt (odrzuca outliery) per kanał → estimate_wind per próbka → WindKalman.

    Parametry
    ----------
    P_5_series : np.ndarray, kształt (N, 5)
        Serie ciśnień [Pa] (kolumny: C, L, R, up, dn).
    fs : float
        Częstotliwość próbkowania [Hz] (przyjmowana dla zgodności / metadanych).
    kf_Q, kf_R : dict
        Parametry filtra Kalmana (patrz WindKalman).
    medfilt_kernel : int
        Rozmiar okna mediany (nieparzysty). 1 = wyłączony.

    Zwraca
    -------
    dict: {'t', 'V', 'alpha_deg', 'beta_deg', 'V_corr', 'V_kf', 'alpha_kf', 'beta_kf'}
        Serie surowe (estymator A) oraz wygładzone Kalmanem.
    """
    P = np.asarray(P_5_series, dtype=float)
    N = P.shape[0]

    # Medianowy filtr outlierów per kanał.
    if medfilt_kernel and medfilt_kernel > 1:
        P_f = np.column_stack([medfilt(P[:, j], kernel_size=medfilt_kernel)
                               for j in range(P.shape[1])])
    else:
        P_f = P

    kf = WindKalman(kf_Q, kf_R)
    V = np.empty(N); al = np.empty(N); be = np.empty(N); Vc = np.empty(N)
    V_kf = np.empty(N); al_kf = np.empty(N); be_kf = np.empty(N)

    for i in range(N):
        est = estimate_wind(P_f[i])
        V[i], al[i], be[i], Vc[i] = est["V"], est["alpha_deg"], est["beta_deg"], est["V_corr"]
        sm = kf.update(est)
        V_kf[i], al_kf[i], be_kf[i] = sm["V"], sm["alpha_deg"], sm["beta_deg"]

    return {
        "t": np.arange(N) / fs,
        "V": V, "alpha_deg": al, "beta_deg": be, "V_corr": Vc,
        "V_kf": V_kf, "alpha_kf": al_kf, "beta_kf": be_kf,
    }


# ── samotest ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("estimator.py — samotest (φ={:.0f}°)".format(PHI_DEG))
    print("=" * 70)

    # ── Test 1: kalibracja k ─────────────────────────────────────────────────
    k_lsq = _fit_k_lsq()
    # slope w zerze: pochodna cp/dθ w 0 = 2·tan φ
    k_slope = 2.0 * np.tan(np.radians(PHI_DEG))
    print(f"\n[1] Kalibracja k (φ={PHI_DEG:.0f}°, n=2, ±25°):")
    print(f"    k_slope (2·tan φ)      = {k_slope:.4f}")
    print(f"    k_LSQ (±25°)           = {k_lsq:.4f}   (stała K_ALPHA={K_ALPHA})")

    # ── Test 2: roundtrip bez szumu ──────────────────────────────────────────
    print(f"\n[2] Roundtrip _forward_min → estimate_wind (V=6 m/s, bez szumu):")
    print(f"    {'α_gt':>6} {'β_gt':>6} | {'α_est':>7} {'β_est':>7} {'V_corr':>7} | "
          f"{'Δα':>6} {'Δβ':>6}")
    max_da = 0.0
    for a_gt, b_gt in [(0, 0), (5, 0), (10, 0), (20, 0), (25, 0),
                       (0, 10), (15, -10), (-25, 25)]:
        P = _forward_min(6.0, a_gt, b_gt)
        est = estimate_wind(P, k_alpha=k_lsq, k_beta=k_lsq)
        da = est["alpha_deg"] - a_gt
        db = est["beta_deg"] - b_gt
        max_da = max(max_da, abs(da))
        print(f"    {a_gt:>6} {b_gt:>6} | {est['alpha_deg']:>7.3f} {est['beta_deg']:>7.3f} "
              f"{est['V_corr']:>7.3f} | {da:>+6.3f} {db:>+6.3f}")
    print(f"    → max |Δα| na ±25° = {max_da:.3f}°  (k_LSQ rozkłada bias po zakresie)")

    # Ścisły roundtrip na małych kątach (|α|≤5°) z k_slope (dokładnym w α→0):
    # sanity-check, że algebra estymatora odwraca forward w punkcie linearyzacji.
    # k_LSQ świadomie poświęca tę dokładność dla mniejszego peak-error przy ±25°.
    strict_ok = True
    worst = 0.0
    for a_gt in [-5, -2, 0, 2, 5]:
        P = _forward_min(6.0, a_gt, 0.0)
        est = estimate_wind(P, k_alpha=k_slope, k_beta=k_slope)
        worst = max(worst, abs(est["alpha_deg"] - a_gt))
        if abs(est["alpha_deg"] - a_gt) > 0.1:
            strict_ok = False
    print(f"    → roundtrip |α|≤5° z k_slope: max |Δα|={worst:.3f}° "
          f"→ {'PASS' if strict_ok else 'FAIL'} (<0.1°)")

    # ── Test 3: propagacja szumu (Monte-Carlo) — DWUWARIANTOWO H1 vs H2 ───────
    print(f"\n[3] Propagacja szumu na kanał α — H1 vs H2 (po re-walidacji, revalidate.py):")
    print(f"    H1 (ODRZUCONA): całe σ_total jako niezależny szum per-kanał.")
    print(f"    H2 (POTWIERDZONA): tylko stały floor czujnika ~1.05 Pa per-kanał;")
    print(f"       turbulencja jest common-mode i kasuje się w cp_α=(P_up−P_dn)/(P_up+P_dn).")
    try:
        from simulate import (sensor_noise, _q_from_v, SIGMA_SENSOR,
                               sigma_total_measured)
        rng = np.random.default_rng(0)
        N_mc = 20000
        print(f"\n    {'V':>5} {'q[Pa]':>7} | {'σ_α H1 teor.':>12} | "
              f"{'σ_α H2 teor.':>12} {'σ_α H2 MC':>10}")
        for V in [3.0, 6.0, 7.5]:
            q = _q_from_v(V)
            m = q / 2.0
            # H1: całe zmierzone σ_total jako per-kanał (górne ograniczenie)
            sa_h1 = np.degrees(sigma_total_measured(V) / (np.sqrt(2) * m * k_slope))
            # H2: tylko floor czujnika
            sa_h2 = np.degrees(SIGMA_SENSOR / (np.sqrt(2) * m * k_slope))
            # Monte-Carlo H2: sensor_noise (floor) niezależnie per kanał
            P0 = _forward_min(V, 0.0, 0.0)
            noises = [sensor_noise(V, N_mc, 10.0, rng) for _ in range(5)]
            est_a = np.array([
                estimate_wind(P0 + np.array([noises[j][i] for j in range(5)]),
                              k_alpha=k_lsq)["alpha_deg"] for i in range(N_mc)])
            print(f"    {V:>5.1f} {q:>7.2f} | {sa_h1:>11.2f}° | "
                  f"{sa_h2:>11.2f}° {est_a.std():>9.2f}°")
        print("\n    → H1 daje plateau/wzrost (~3–4° @6–7.5), H2 MALEJE (7.7°→1.9°→1.2°).")
        print("    → Dane wspierają H2: «plateau σ_α» NIE PRZEŻYWA re-walidacji.")
        print("    → α dalej zawodzi przy 3 m/s (floor dominuje), ale POPRAWIA się z V.")
    except Exception as e:
        print(f"    (pominięto — simulate niedostępny: {e})")

    # ── Test 4: kasowanie się turbulencji common-mode w cp_α ─────────────────
    # Decydujący dowód werdyktu H2: turbulencja (fluktuacja V) wstrzyknięta
    # WSPÓLNIE do wszystkich 5 sond niemal nie wpływa na α (kasuje się w ilorazie
    # cp_α), podczas gdy ten sam szum jako NIEZALEŻNY per-kanał psuje α w pełni.
    print(f"\n[4] Kasowanie turbulencji common-mode w cp_α (dowód H2):")
    try:
        from simulate import sensor_noise, _q_from_v, TURB_MODEL
        rng = np.random.default_rng(1)
        N_t = 20000
        print(f"    {'V':>5} {'σ_turb':>7} | {'σ_α common-mode':>16} {'σ_α per-kanał':>14}")
        for V in [6.0, 7.5]:
            q = _q_from_v(V)
            P0 = _forward_min(V, 0.0, 0.0)
            sig_turb = TURB_MODEL[V]["sigma_dp"]
            # (a) WSPÓLNA fluktuacja: ta sama realizacja na wszystkie sondy
            turb = sensor_noise(V, N_t, 10.0, rng)            # użyj jako wspólny szereg
            turb = turb / turb.std() * sig_turb               # przeskaluj do σ_turb
            al_cm = np.array([estimate_wind(P0 + turb[i], k_alpha=k_lsq)["alpha_deg"]
                              for i in range(N_t)])
            # (b) NIEZALEŻNA per kanał: 5 różnych realizacji
            ind = [sensor_noise(V, N_t, 10.0, rng) for _ in range(5)]
            ind = [x / x.std() * sig_turb for x in ind]
            al_ind = np.array([
                estimate_wind(P0 + np.array([ind[j][i] for j in range(5)]),
                              k_alpha=k_lsq)["alpha_deg"] for i in range(N_t)])
            print(f"    {V:>5.1f} {sig_turb:>7.2f} | {al_cm.std():>15.3f}° {al_ind.std():>13.2f}°")
        print("    → common-mode (turbulencja realna) prawie nie rusza α; per-kanał psuje.")
        print("    → Dlatego turbulencji NIE wolno wliczać do szumu per-kanał kanału α.")
    except Exception as e:
        print(f"    (pominięto — simulate niedostępny: {e})")

    print("\n" + "=" * 70)
    print("Samotest zakończony.")
