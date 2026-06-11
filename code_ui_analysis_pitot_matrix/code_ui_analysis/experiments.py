"""
experiments.py — macierz eksperymentów degradacyjnych estymatora A (dane SYMULOWANE).

Główny produkt to KRZYWE DEGRADACJI: jak błąd estymacji (V, α, β) rośnie pod
kontrolowanymi zaburzeniami. Każdy punkt = Monte-Carlo (≥300 prób).

Potok jednej próby:
  GT (V,α,β) → physics.forward_5probe (BOGATY generator) → +szum floor per kanał
  + kwantyzacja ADC → estimator.estimate_wind (wariant A, zakłada n=2) → błąd.

Generator jest celowo bogatszy niż estymator (inverse crime) — różnica to wynik.
Wyniki zapisywane do results/*.json; wykresy rysuje plotting.py.
"""

import os, json
import numpy as np

from physics import forward_5probe, RHO_REF
from estimator import estimate_wind, _fit_k_lsq, WindKalman
from simulate import SIGMA_SENSOR, LSB_DP, _colored_noise_fft, TURB_MODEL

RESULTS_DIR = "results"
N_MC = 400               # prób Monte-Carlo na punkt (≥300)
K_CAL = _fit_k_lsq()     # k_α=k_β skalibrowane dla n=2, φ=45°, ±25°

# Konfiguracja generatora — "czysta baza" (n=2 zgodne z estymatorem), tak by każdy
# sweep izolował WPŁYW JEDNEJ zmiennej. Zaburzenia włączane per eksperyment.
BASE_GEN = dict(n=2.0, crosstalk_dn=1.0, gain=None, bias_pa=None)

SPEEDS = [3.0, 6.0, 7.5]
ALPHA_GRID = [-20.0, -10.0, 0.0, 10.0, 20.0]   # reprezentatywne kąty do RMSE


# ── narzędzia Monte-Carlo ─────────────────────────────────────────────────────

def _draw_noise(sigma, lsb, rng, size=5):
    """Biały szum czujnika per kanał [Pa] + kwantyzacja ADC (LSB)."""
    x = rng.normal(0.0, sigma, size=size)
    if lsb and lsb > 0:
        x = np.round(x / lsb) * lsb
    return x


def _vcorr_guard(est, V_true):
    """
    Zabezpieczona korekta V (estymatora nie zmieniamy — guard liczony tu).

    V_corr = V_raw/cos(θ) eksploduje, gdy przy niskim SNR kąt jest niewiarygodny
    (θ→90° ⇒ cos→0). Ograniczamy θ≤60° i cos≥0.5 (korekta max ×2). Bez tego
    pojedyncze outliery dominują RMSE(V) przy 3 m/s — to znana niestabilność,
    nie chcemy jej maskować, ale i nie chcemy by zaśmiecała statystykę.
    """
    th = np.radians(np.hypot(est["alpha_deg"], est["beta_deg"]))
    cosg = max(np.cos(min(th, np.radians(60.0))), 0.5)
    return est["V"] / cosg - V_true


def mc_point(V, alpha, beta, gen_kw, sigma, lsb, rng,
             n_mc=N_MC, common_turb=0.0, perchan_turb=0.0):
    """
    RMSE (V_corr, α, β) dla jednego GT przez Monte-Carlo.

    common_turb : σ [Pa] turbulencji COMMON-MODE — fluktuacja prędkości, modelowana
                  MULTIPLIKATYWNIE: wszystkie q skalują się tym samym czynnikiem
                  (1+ε), ε~N(0, common_turb/q). Faktoryzuje się i KASUJE w cp.
    perchan_turb: σ [Pa] tej samej fluktuacji, ale NIEZALEŻNEJ per sonda (nie kasuje).
    Zwraca: (rmse_V, rmse_a, rmse_b, bias_a) — bias_a = średni błąd α (systematyczny).
    """
    P0 = forward_5probe(V, alpha, beta, **gen_kw)
    q = 0.5 * RHO_REF * V ** 2
    eV = np.empty(n_mc); ea = np.empty(n_mc); eb = np.empty(n_mc)
    for i in range(n_mc):
        P = P0 + _draw_noise(sigma, lsb, rng, 5)
        if common_turb > 0:               # multiplikatywna, wspólna → kasuje się w cp
            P = P * (1.0 + rng.normal(0.0, common_turb / q))
        if perchan_turb > 0:              # multiplikatywna, niezależna per sonda
            P = P * (1.0 + rng.normal(0.0, perchan_turb / q, size=5))
        est = estimate_wind(P, k_alpha=K_CAL, k_beta=K_CAL)
        eV[i] = _vcorr_guard(est, V)
        ea[i] = est["alpha_deg"] - alpha
        eb[i] = est["beta_deg"] - beta
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    return rmse(eV), rmse(ea), rmse(eb), float(np.mean(ea))


def mc_over_grid(V, gen_kw, sigma, lsb, rng, alphas=ALPHA_GRID, beta=0.0,
                 n_mc=N_MC, common_turb=0.0, perchan_turb=0.0):
    """RMSE uśrednione (RMS) po siatce kątów α — reprezentatywne dla prędkości V."""
    rV, ra, rb = [], [], []
    for a in alphas:
        v, x, b, _ = mc_point(V, a, beta, gen_kw, sigma, lsb, rng, n_mc,
                              common_turb, perchan_turb)
        rV.append(v); ra.append(x); rb.append(b)
    rms = lambda lst: float(np.sqrt(np.mean(np.square(lst))))
    return rms(rV), rms(ra), rms(rb)


# ── eksperymenty ──────────────────────────────────────────────────────────────

def exp_rmse_vs_speed(rng):
    """E1: RMSE(V,α,β) vs prędkość (baza czysta n=2; izoluje wpływ SNR → H2).

    Siatka umiarkowanych kątów (|α|≤10°), by uchwycić limit SNR bez nasycenia
    arcsin przy dużych kątach (degradacja dużych kątów: osobno E9)."""
    mod_grid = [-10.0, -5.0, 0.0, 5.0, 10.0]
    out = {"speed": SPEEDS, "rmse_V": [], "rmse_a": [], "rmse_b": []}
    for V in SPEEDS:
        rV, ra, rb = mc_over_grid(V, BASE_GEN, SIGMA_SENSOR, LSB_DP, rng, alphas=mod_grid)
        out["rmse_V"].append(rV); out["rmse_a"].append(ra); out["rmse_b"].append(rb)
    return out


def exp_vs_noise(rng):
    """E2: RMSE(α) vs poziom szumu czujnika (σ wokół 1.05 Pa), V=3 i 7.5."""
    sigmas = list(np.round(np.linspace(0.2, 3.0, 11), 3))
    out = {"sigma": sigmas, "V": [3.0, 7.5], "rmse_a": {}}
    for V in out["V"]:
        ra = [mc_over_grid(V, BASE_GEN, s, LSB_DP, rng)[1] for s in sigmas]
        out["rmse_a"][str(V)] = ra
    return out


def exp_vs_quant(rng):
    """E3: RMSE(α) vs siła kwantyzacji LSB, V=3 i 7.5 (kluczowe przy niskim V)."""
    lsbs = list(np.round(np.linspace(0.0, 1.0, 11), 3))
    out = {"lsb": lsbs, "V": [3.0, 7.5], "rmse_a": {}}
    for V in out["V"]:
        ra = [mc_over_grid(V, BASE_GEN, SIGMA_SENSOR, l, rng)[1] for l in lsbs]
        out["rmse_a"][str(V)] = ra
    return out


def exp_vs_cosn(rng):
    """E4: RMSE(α) vs niedopasowanie cos^n (generator n, estymator zakłada n=2)."""
    ns = [2.0, 2.25, 2.5, 2.75, 3.0]
    out = {"n": ns, "V": 6.0, "rmse_a": [], "bias_a_at20": []}
    for n in ns:
        gk = dict(BASE_GEN); gk["n"] = n
        _, ra, _ = mc_over_grid(6.0, gk, SIGMA_SENSOR, LSB_DP, rng)
        # bias systematyczny przy dużym kącie (α=20°)
        _, _, _, bias = mc_point(6.0, 20.0, 0.0, gk, SIGMA_SENSOR, LSB_DP, rng)
        out["rmse_a"].append(ra); out["bias_a_at20"].append(bias)
    return out


def exp_vs_bias(rng):
    """E5: RMSE(α) vs amplituda biasu montażowego sond (asymetria wzmocnień), V=6."""
    amps = list(np.round(np.linspace(0.0, 0.05, 11), 4))   # σ wzmocnienia per sonda
    out = {"bias_amp": amps, "V": 6.0, "rmse_a": []}
    for amp in amps:
        ras = []
        for _ in range(8):   # uśrednij po realizacjach losowego biasu
            g = 1.0 + rng.normal(0.0, amp, size=5)
            gk = dict(BASE_GEN); gk["gain"] = g
            _, ra, _ = mc_over_grid(6.0, gk, SIGMA_SENSOR, LSB_DP, rng, n_mc=150)
            ras.append(ra)
        out["rmse_a"].append(float(np.mean(ras)))
    return out


def exp_common_mode(rng):
    """E6 (KLUCZOWY dowód H2): RMSE(α) vs amplituda turbulencji — common-mode vs per-kanał."""
    amps = list(np.round(np.linspace(0.0, 5.0, 11), 3))
    out = {"turb_amp": amps, "V": 6.0, "rmse_common": [], "rmse_perchan": []}
    for amp in amps:
        # (a) common-mode: multiplikatywna fluktuacja prędkości WSPÓLNA dla sond
        _, ra_cm, _ = mc_over_grid(6.0, BASE_GEN, SIGMA_SENSOR, LSB_DP, rng,
                                   common_turb=amp)
        # (b) per-kanał: ta sama amplituda, ale NIEZALEŻNA per sonda (nie kasuje)
        _, ra_pc, _ = mc_over_grid(6.0, BASE_GEN, SIGMA_SENSOR, LSB_DP, rng,
                                   perchan_turb=amp)
        out["rmse_common"].append(ra_cm)
        out["rmse_perchan"].append(ra_pc)
    return out


def exp_filtering(rng):
    """E7: surowa vs filtracja czasowa (Kalman/boxcar), szum biały (floor) — redukcja."""
    out = {"V": SPEEDS, "sigma_raw": [], "sigma_filt": [], "reduction": []}
    N = 4000
    for V in SPEEDS:
        P0 = forward_5probe(V, 0.0, 0.0, **BASE_GEN)
        # 5 niezależnych białych szeregów floor
        noises = []
        for _ in range(5):
            x = _colored_noise_fft(N, -0.1, rng) * SIGMA_SENSOR
            x = np.round(x / LSB_DP) * LSB_DP
            noises.append(x)
        al = np.array([estimate_wind(P0 + np.array([noises[j][i] for j in range(5)]),
                                     k_alpha=K_CAL)["alpha_deg"] for i in range(N)])
        M = 20  # boxcar 2 s @10 Hz ~ silna filtracja
        al_f = np.convolve(al, np.ones(M) / M, mode="valid")
        out["sigma_raw"].append(float(al.std()))
        out["sigma_filt"].append(float(al_f.std()))
        out["reduction"].append(float(al.std() / al_f.std()))
    return out


def exp_error_maps(rng):
    """E8: mapy błędu RMSE(α) w przestrzeni (α,β) dla 3 prędkości."""
    grid = list(np.linspace(-25, 25, 11))
    out = {"alpha": grid, "beta": grid, "V": SPEEDS, "maps": {}}
    for V in SPEEDS:
        M = np.zeros((len(grid), len(grid)))
        for ia, a in enumerate(grid):
            for ib, b in enumerate(grid):
                _, ra, _, _ = mc_point(V, a, b, BASE_GEN, SIGMA_SENSOR, LSB_DP,
                                       rng, n_mc=150)
                M[ia, ib] = ra
        out["maps"][str(V)] = M.tolist()
    return out


def exp_angle_breakdown(rng):
    """E9: rozbicie RMSE(α) na małe (|α|≤10°) vs duże (|α|≥20°) kąty, per prędkość."""
    out = {"speed": SPEEDS, "small": [], "large": []}
    small_a = [-10, -5, 0, 5, 10]
    large_a = [-25, -22, 22, 25]
    for V in SPEEDS:
        _, rs, _ = mc_over_grid(V, BASE_GEN, SIGMA_SENSOR, LSB_DP, rng, alphas=small_a)
        _, rl, _ = mc_over_grid(V, BASE_GEN, SIGMA_SENSOR, LSB_DP, rng, alphas=large_a)
        out["small"].append(rs); out["large"].append(rl)
    return out


# ── orkiestracja ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(2026)
    print(f"experiments.py — macierz degradacji (N_MC={N_MC}, k_cal={K_CAL:.4f})")
    print("=" * 64)

    registry = [
        ("rmse_vs_speed", exp_rmse_vs_speed),
        ("vs_noise",      exp_vs_noise),
        ("vs_quant",      exp_vs_quant),
        ("vs_cosn",       exp_vs_cosn),
        ("vs_bias",       exp_vs_bias),
        ("common_mode",   exp_common_mode),
        ("filtering",     exp_filtering),
        ("error_maps",    exp_error_maps),
        ("angle_breakdown", exp_angle_breakdown),
    ]
    results = {"meta": {"N_MC": N_MC, "k_cal": K_CAL, "sigma_sensor": SIGMA_SENSOR,
                        "lsb": LSB_DP, "speeds": SPEEDS}}
    for name, fn in registry:
        print(f"  [{name}] ...", end="", flush=True)
        results[name] = fn(rng)
        print(" ok")

    path = os.path.join(RESULTS_DIR, "experiments.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nZapisano {path}")

    # ── kluczowe liczby ──────────────────────────────────────────────────────
    r = results
    print("\nKLUCZOWE LICZBY:")
    print(f"  RMSE(α) vs V:  " + ", ".join(
        f"{V}m/s={ra:.2f}°" for V, ra in zip(r['rmse_vs_speed']['speed'],
                                             r['rmse_vs_speed']['rmse_a'])))
    print(f"  RMSE(V) vs V:  " + ", ".join(
        f"{V}m/s={rv:.3f}m/s" for V, rv in zip(r['rmse_vs_speed']['speed'],
                                               r['rmse_vs_speed']['rmse_V'])))
    cm = r["common_mode"]
    print(f"  Common-mode @V=6: turb 0→5 Pa: RMSE_α {cm['rmse_common'][0]:.2f}° → "
          f"{cm['rmse_common'][-1]:.2f}° (PŁASKO = kasuje się)")
    print(f"               per-kanał:           RMSE_α {cm['rmse_perchan'][0]:.2f}° → "
          f"{cm['rmse_perchan'][-1]:.2f}° (rośnie)")
    cn = r["vs_cosn"]
    print(f"  cos^n n=2→3 @V=6: RMSE_α {cn['rmse_a'][0]:.2f}° → {cn['rmse_a'][-1]:.2f}°; "
          f"bias@20° {cn['bias_a_at20'][0]:+.2f}° → {cn['bias_a_at20'][-1]:+.2f}°")
    ft = r["filtering"]
    print(f"  Filtracja redukcja σ_α: " + ", ".join(
        f"{V}m/s ×{rd:.1f}" for V, rd in zip(ft['V'], ft['reduction'])))
    return results


if __name__ == "__main__":
    main()
