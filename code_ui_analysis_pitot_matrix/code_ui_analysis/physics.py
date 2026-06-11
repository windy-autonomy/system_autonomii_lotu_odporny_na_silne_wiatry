"""
physics.py — model fizyczny matrycy 5 rurek Pitota (GENERATOR „prawdy").

To jest GENERATOR danych syntetycznych, NIE estymator. Zgodnie z zasadą unikania
inverse crime generator jest CELOWO BOGATSZY niż estymator A (estimator.py):

  efekt                      | generator (tu)        | estymator A (zakłada)
  ---------------------------|-----------------------|----------------------
  wykładnik kosinusa cos^n   | n konfigurowalne (2–3)| zawsze n = 2
  kantowanie sond φ          | konfigurowalne        | zaszyte w k_α (φ=45°)
  bias montażowy per sonda   | TAK (wzmocnienie g_i) | brak
  crosstalk sondy dolnej     | TAK (~0.97)           | brak
  sprzężenie płaszczyzn α–β  | TAK (człon cos)       | brak

Różnica między prawdą (generator) a estymatą A pochodzi z efektów, których A NIE
modeluje — i to JEST wynik naukowy (krzywe degradacji), nie defekt.

Kolejność sond w wektorze P_5: (C, L, R, up, dn) — zgodna z estimator.IDX.

φ = 45° to ZAŁOŻENIE PROJEKTOWE (nie zmierzony kąt) — patrz CLAUDE.md / raport.
"""

import numpy as np

# Kolejność sond — MUSI być zgodna z estimator.IDX = dict(C=0,L=1,R=2,UP=3,DN=4)
PROBE_ORDER = ("C", "L", "R", "UP", "DN")
IDX = {name: i for i, name in enumerate(PROBE_ORDER)}

RHO_REF = 1.225      # kg/m³ (ISA, poziom morza)
R_AIR = 287.05       # J/(kg·K)
PHI_DEG = 45.0       # kąt kantowania sond [deg] — ZAŁOŻENIE


def air_density(T_C: float = 15.0, p_atm: float = 101325.0) -> float:
    """
    Gęstość powietrza z równania stanu gazu doskonałego: ρ = p / (R·T).

    T_C : temperatura [°C], p_atm : ciśnienie atmosferyczne [Pa].
    Domyślnie ISA poziom morza → ρ ≈ 1.225 kg/m³.
    """
    return p_atm / (R_AIR * (T_C + 273.15))


def probe_dynamic_pressure(V, theta_inflow, rho=RHO_REF, n=2.0):
    """
    Ciśnienie dynamiczne sondy pod kątem napływu θ: q = 0.5·ρ·V²·cos^n(θ).

    cos zaciskane do ≥0 (sonda „za" przepływem nie generuje podciśnienia w modelu).
    n = wykładnik kosinusowy (2 = teoria idealna; 2.5–3 = realny spadek czulości).
    """
    c = max(np.cos(theta_inflow), 0.0)
    return 0.5 * rho * V ** 2 * c ** n


def forward_5probe(V, alpha_deg, beta_deg, n=2.0, phi_deg=PHI_DEG,
                   gain=None, crosstalk_dn=1.0, bias_pa=None, rho=RHO_REF):
    """
    Bogaty forward model krzyża 5 sond → wektor 5 ciśnień [Pa] (C, L, R, up, dn).

    Model kątów efektywnych (separowalny, z lekkim sprzężeniem płaszczyzn przez
    człon cos drugiej osi — efekt, którego estymator A nie modeluje):
        cosθ_C  = cosα·cosβ
        cosθ_up = cos(φ−α)·cosβ      cosθ_dn = cos(φ+α)·cosβ
        cosθ_R  = cosα·cos(φ−β)      cosθ_L  = cosα·cos(φ+β)
    q_i = 0.5·ρ·V²·cosθ_i^n.

    Parametry zaburzeń (łamią inverse crime):
    ----------
    n : float
        Wykładnik kosinusa w GENERATORZE. n=2 → zgodny z estymatorem (brak
        niedopasowania); n∈{2.5,3} → realny spadek czułości, estymator (n=2)
        wprowadza błąd systematyczny rosnący z |kąt|.
    phi_deg : float
        Kąt kantowania sond [deg].
    gain : array-like (5,) lub None
        Multiplikatywny bias montażowy/kalibracyjny per sonda (g_i, domyślnie 1.0).
        Asymetria g_up≠g_dn tworzy błąd α nawet przy α=0.
    crosstalk_dn : float
        Przeciek pneumatyczny sondy dolnej (domyślnie 1.0 = brak; realnie ~0.97).
    bias_pa : array-like (5,) lub None
        Addytywny offset ciśnienia per sonda [Pa] (np. dryf zera czujnika).
    rho : float
        Gęstość powietrza [kg/m³].

    Zwraca: np.ndarray (5,) ciśnień [Pa] w kolejności PROBE_ORDER.
    """
    a = np.radians(alpha_deg)
    b = np.radians(beta_deg)
    phi = np.radians(phi_deg)
    q = 0.5 * rho * V ** 2

    def cn(x):
        return max(np.cos(x), 0.0) ** n

    cb = max(np.cos(b), 0.0)
    ca = max(np.cos(a), 0.0)

    P = np.empty(5)
    P[IDX["C"]]  = q * (ca * cb) ** n
    P[IDX["UP"]] = q * (max(np.cos(phi - a), 0.0) * cb) ** n
    P[IDX["DN"]] = q * (max(np.cos(phi + a), 0.0) * cb) ** n
    P[IDX["R"]]  = q * (ca * max(np.cos(phi - b), 0.0)) ** n
    P[IDX["L"]]  = q * (ca * max(np.cos(phi + b), 0.0)) ** n

    # Crosstalk sondy dolnej (przeciek pneumatyczny).
    P[IDX["DN"]] *= crosstalk_dn

    # Bias montażowy/kalibracyjny multiplikatywny per sonda.
    if gain is not None:
        P = P * np.asarray(gain, dtype=float)

    # Addytywny offset zera per sonda.
    if bias_pa is not None:
        P = P + np.asarray(bias_pa, dtype=float)

    return P


# ── szybki test roundtrip (generator n=2, bez zaburzeń → estymator A) ──────────

if __name__ == "__main__":
    from estimator import estimate_wind, _fit_k_lsq
    k = _fit_k_lsq()
    print("physics.py — sanity forward_5probe → estimate_wind (n=2, bez zaburzeń):")
    print(f"  ρ(15°C, 101325 Pa) = {air_density():.4f} kg/m³")
    print(f"  {'V':>4} {'α_gt':>5} {'β_gt':>5} | {'V_est':>6} {'α_est':>6} {'β_est':>6}")
    for V, a, b in [(6, 0, 0), (6, 10, 0), (6, 0, 10), (7.5, 20, -15), (3, -25, 25)]:
        P = forward_5probe(V, a, b, n=2.0)
        e = estimate_wind(P, k_alpha=k, k_beta=k)
        print(f"  {V:>4} {a:>5} {b:>5} | {e['V']:>6.2f} {e['alpha_deg']:>6.2f} {e['beta_deg']:>6.2f}")

    print("\n  Zaburzenia łamiące inverse crime (V=6, α=15°, β=0):")
    P0 = forward_5probe(6, 15, 0, n=2.0)
    for label, kw in [("n=3 (cos^n)", dict(n=3.0)),
                      ("crosstalk dn=0.97", dict(n=2.0, crosstalk_dn=0.97)),
                      ("gain bias up+3%", dict(n=2.0, gain=[1, 1, 1, 1.03, 1]))]:
        P = forward_5probe(6, 15, 0, **kw)
        e = estimate_wind(P, k_alpha=k, k_beta=k)
        print(f"    {label:22s} → α_est={e['alpha_deg']:6.2f}°  (Δ={e['alpha_deg']-15:+.2f}°)")
