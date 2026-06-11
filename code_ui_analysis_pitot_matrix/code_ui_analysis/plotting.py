"""
plotting.py — wykresy pod raport SKNTI (matryca Pitota, część eksperymentalno-wynikowa).

Styl spójny z make_pitot_rmse.py: polskie podpisy, osie z jednostkami, paleta SKNTI,
PNG ≥150 DPI, każdy wykres samodzielny. Czyta results/experiments.json, zapisuje do
SKNTI/img/.

Lista figur:
  P1 diagram algorytmu A          → alg_diagram.png
  P2 RMSE(α) vs prędkość          → pitot_rmse_vs_speed.png   (nadpisuje istniejący)
  P2b RMSE(V, α, β) razem         → pitot_rmse_all.png
  P3 krzywe degradacji (panel 2×2)→ degradation_panel.png
  P4 dowód kasowania common-mode  → common_mode_proof.png
  P5 mapy błędu (α,β) 3 prędkości → error_maps_ab.png
  P6 surowa vs filtrowana         → raw_vs_filtered.png
  P7 rozbicie małe/duże kąty      → angle_breakdown.png
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── styl ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "font.family": "DejaVu Sans",
})
GREEN, RED, BLUE, AMBER, VIOLET = "#1D9E75", "#E24B4A", "#378ADD", "#BA7517", "#7B4FB7"
GRAY = "#666666"

IMG_DIR = "SKNTI/img"
RESULTS = "results/experiments.json"


def _load():
    with open(RESULTS) as f:
        return json.load(f)


def _save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  zapisano {path}")


# ── P1: diagram algorytmu A ───────────────────────────────────────────────────

def plot_alg_diagram():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    linewidth=1.6, edgecolor=color,
                                    facecolor=color + "22"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)

    def arrow(x0, y0, x1, y1, color=GRAY):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=14, linewidth=1.4, color=color))

    box(1, 17, 13, 12, "5 sond Pitota\n$P_C,P_L,P_R,P_{up},P_{dn}$", BLUE)
    box(18, 17, 13, 12, "filtr medianowy\n(outliery)", GRAY)
    box(35, 28, 16, 12, "$V=\\sqrt{2\\Delta p_C/\\rho}$", GREEN)
    box(35, 15, 16, 11, "$cp_\\alpha,\\,cp_\\beta$\nróżnice par", GREEN)
    box(35, 2, 16, 11, "$\\alpha,\\beta=\\arcsin(cp/k)$", GREEN)
    box(55, 15, 15, 12, "korekta\n$V_{corr}=V/\\cos\\theta$", AMBER)
    box(73, 15, 12, 12, "filtr\nKalmana 1D", VIOLET)
    box(88, 15, 11, 12, "kierunek\nwiatru", RED)

    arrow(14, 23, 18, 23); arrow(31, 23, 35, 34); arrow(31, 23, 35, 20.5)
    arrow(31, 22, 35, 7.5)
    arrow(51, 34, 62.5, 27)            # V → korekta
    arrow(51, 7.5, 62.5, 15)           # α,β → korekta
    arrow(70, 21, 73, 21)              # korekta → Kalman
    arrow(85, 21, 88, 21)              # Kalman → wiatr
    ax.text(50, 43.5, "Algorytm estymacji A (różnice par)", ha="center",
            fontsize=12, weight="bold")
    ax.text(91.5, 9.5, "(po złożeniu z\norientacją modelu)", ha="center",
            fontsize=7.5, color=GRAY)
    _save(fig, "alg_diagram.png")


# ── P2: RMSE(α) vs prędkość (surowa + filtrowana) ─────────────────────────────

def plot_rmse_vs_speed(r):
    sp = np.array(r["rmse_vs_speed"]["speed"])
    raw = np.array(r["rmse_vs_speed"]["rmse_a"])
    red = np.array(r["filtering"]["reduction"])
    filt = raw / red

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(sp, raw, "o--", color=RED, lw=1.8, ms=9, label="estymacja surowa")
    ax.plot(sp, filt, "o-", color=GREEN, lw=2.2, ms=10, label="po filtracji czasowej")
    for V, a, b in zip(sp, raw, filt):
        ax.annotate(f"{a:.1f}°", (V, a), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color=RED)
        ax.annotate(f"{b:.1f}°", (V, b), textcoords="offset points",
                    xytext=(8, -14), fontsize=9, color="#0F6E56")
    ax.set_xlabel("prędkość napływu  V  [m/s]")
    ax.set_ylabel("błąd estymacji kąta natarcia  RMSE(α)  [°]")
    ax.set_title("Dokładność estymacji kąta natarcia w funkcji prędkości\n"
                 "(szum czujnika stały ~1,05 Pa; SNR rośnie z prędkością)")
    ax.set_xticks(sp); ax.set_ylim(bottom=0); ax.legend(loc="upper right")
    ax.axvspan(2.5, 4.0, alpha=0.06, color=RED)
    ax.text(3.0, ax.get_ylim()[1] * 0.55, "niski SNR\n+ kwantyzacja", ha="center",
            fontsize=8.5, color="#A03028")
    _save(fig, "pitot_rmse_vs_speed.png")


# ── P2b: RMSE(V, α, β) — trzy kanały obok siebie ─────────────────────────────

def plot_rmse_all_channels(r):
    sp = np.array(r["rmse_vs_speed"]["speed"])
    rmse_a = np.array(r["rmse_vs_speed"]["rmse_a"])
    rmse_b = np.array(r["rmse_vs_speed"]["rmse_b"])
    rmse_V = np.array(r["rmse_vs_speed"]["rmse_V"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    pairs = [
        (axes[0], rmse_a, "RMSE(α)  [°]",    "kąt natarcia  α",  RED),
        (axes[1], rmse_b, "RMSE(β)  [°]",    "kąt znoszenia  β", BLUE),
        (axes[2], rmse_V, "RMSE(V)  [m/s]",  "prędkość  V",      GREEN),
    ]
    for ax, vals, ylabel, title, color in pairs:
        ax.plot(sp, vals, "o-", color=color, lw=2.2, ms=9)
        for V, v in zip(sp, vals):
            ax.annotate(f"{v:.2f}", (V, v), textcoords="offset points",
                        xytext=(6, 6), fontsize=9, color=color)
        ax.set_xlabel("prędkość napływu  V  [m/s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(sp)
        ax.set_ylim(bottom=0)

    fig.suptitle("Błąd estymacji wszystkich kanałów w funkcji prędkości napływu "
                 "(dane symulowane, szum czujnika ~1,05 Pa)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "pitot_rmse_all.png")


# ── P3: panel degradacji 2×2 ──────────────────────────────────────────────────

def plot_degradation_panel(r):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # (a) szum czujnika
    n = r["vs_noise"]
    for V, c in zip(n["V"], (RED, GREEN)):
        ax[0, 0].plot(n["sigma"], n["rmse_a"][str(V)], "o-", color=c, lw=1.8, ms=5,
                      label=f"V={V} m/s")
    ax[0, 0].axvline(r["meta"]["sigma_sensor"], color=GRAY, ls=":", lw=1.2)
    ax[0, 0].text(r["meta"]["sigma_sensor"], ax[0, 0].get_ylim()[1] * 0.9,
                  " floor≈1.05 Pa", fontsize=8, color=GRAY)
    ax[0, 0].set_xlabel("σ szumu czujnika  [Pa]")
    ax[0, 0].set_ylabel("RMSE(α)  [°]")
    ax[0, 0].set_title("(a) wpływ szumu czujnika")
    ax[0, 0].legend()

    # (b) kwantyzacja
    q = r["vs_quant"]
    for V, c in zip(q["V"], (RED, GREEN)):
        ax[0, 1].plot(q["lsb"], q["rmse_a"][str(V)], "o-", color=c, lw=1.8, ms=5,
                      label=f"V={V} m/s")
    ax[0, 1].axvline(r["meta"]["lsb"], color=GRAY, ls=":", lw=1.2)
    ax[0, 1].text(r["meta"]["lsb"], ax[0, 1].get_ylim()[1] * 0.9, " LSB=0.12 Pa",
                  fontsize=8, color=GRAY)
    ax[0, 1].set_xlabel("krok kwantyzacji  LSB  [Pa]")
    ax[0, 1].set_ylabel("RMSE(α)  [°]")
    ax[0, 1].set_title("(b) wpływ kwantyzacji ADC")
    ax[0, 1].legend()

    # (c) niedopasowanie cos^n
    cn = r["vs_cosn"]
    ax[1, 0].plot(cn["n"], cn["rmse_a"], "o-", color=VIOLET, lw=2.0, ms=7,
                  label="RMSE(α)")
    ax[1, 0].plot(cn["n"], np.abs(cn["bias_a_at20"]), "s--", color=AMBER, lw=1.6,
                  ms=6, label="|bias| przy α=20°")
    ax[1, 0].set_xlabel("wykładnik cos^n w generatorze  (estymator zakłada n=2)")
    ax[1, 0].set_ylabel("błąd α  [°]")
    ax[1, 0].set_title("(c) niedopasowanie modelu cos^n  (V=6 m/s)")
    ax[1, 0].legend()

    # (d) bias montażowy
    b = r["vs_bias"]
    ax[1, 1].plot(np.array(b["bias_amp"]) * 100, b["rmse_a"], "o-", color=BLUE,
                  lw=2.0, ms=6)
    ax[1, 1].set_xlabel("amplituda biasu montażowego sond  [% wzmocnienia]")
    ax[1, 1].set_ylabel("RMSE(α)  [°]")
    ax[1, 1].set_title("(d) wpływ biasu montażowego  (V=6 m/s)")

    fig.suptitle("Krzywe degradacji estymatora A pod kontrolowanymi zaburzeniami "
                 "(dane symulowane)", fontsize=13, y=0.995)
    fig.tight_layout()
    _save(fig, "degradation_panel.png")


# ── P4: dowód kasowania common-mode ───────────────────────────────────────────

def plot_common_mode(r):
    c = r["common_mode"]
    fig, ax = plt.subplots(figsize=(7.8, 5))
    ax.plot(c["turb_amp"], c["rmse_perchan"], "o-", color=RED, lw=2.0, ms=6,
            label="szum NIEZALEŻNY per sonda")
    ax.plot(c["turb_amp"], c["rmse_common"], "o-", color=GREEN, lw=2.4, ms=7,
            label="turbulencja COMMON-MODE (wspólna)")
    ax.set_xlabel("amplituda fluktuacji przepływu  [Pa]")
    ax.set_ylabel("RMSE(α)  [°]")
    ax.set_title("Dowód kasowania zakłócenia wspólnego w różnicy par\n"
                 "(turbulencja strugi nie psuje estymacji α)  V=6 m/s")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    ax.annotate("kasuje się w $cp_\\alpha$", (c["turb_amp"][-1], c["rmse_common"][-1]),
                textcoords="offset points", xytext=(-40, 14), fontsize=9, color="#0F6E56")
    _save(fig, "common_mode_proof.png")


# ── P5: mapy błędu (α,β) ──────────────────────────────────────────────────────

def plot_error_maps(r):
    em = r["error_maps"]
    a = np.array(em["alpha"]); b = np.array(em["beta"])
    speeds = em["V"]
    vmax = max(np.max(em["maps"][str(V)]) for V in speeds)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, V in zip(axes, speeds):
        M = np.array(em["maps"][str(V)])
        im = ax.pcolormesh(b, a, M, cmap="inferno", vmin=0, vmax=vmax, shading="auto")
        ax.set_title(f"V = {V} m/s")
        ax.set_xlabel("kąt znoszenia  β  [°]")
        ax.grid(False)
        if ax is axes[0]:
            ax.set_ylabel("kąt natarcia  α  [°]")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("RMSE(α)  [°]")
    fig.suptitle("Mapy błędu estymacji kąta natarcia w przestrzeni (α, β) — dane symulowane",
                 fontsize=13)
    _save(fig, "error_maps_ab.png")


# ── P6: surowa vs filtrowana (przebieg na profilu schodkowym) ─────────────────

def plot_raw_vs_filtered():
    from physics import forward_5probe
    from estimator import estimate_wind, WindKalman, _fit_k_lsq
    from simulate import sensor_noise, SIGMA_SENSOR, step_alpha_profile

    rng = np.random.default_rng(7)
    fs, V = 10.0, 3.0           # 3 m/s — najtrudniejszy przypadek (biały floor)
    k = _fit_k_lsq()
    alpha_gt = step_alpha_profile(fs, [-20, -10, 0, 10, 20, 0], dwell_s=15)
    N = len(alpha_gt)
    noises = [sensor_noise(V, N, fs, rng) for _ in range(5)]

    al_raw = np.empty(N)
    for i in range(N):
        P = forward_5probe(V, alpha_gt[i], 0.0, n=2.0)
        P = P + np.array([noises[j][i] for j in range(5)])
        al_raw[i] = estimate_wind(P, k_alpha=k)["alpha_deg"]

    # Kalman 1D: R z propagacji (~σ_α²), Q małe (profil quasi-statyczny)
    R_a = (np.degrees(SIGMA_SENSOR / (np.sqrt(2) * (0.5 * 1.225 * V ** 2 / 2) * 2.0))) ** 2
    kf = WindKalman(Q={"V": 1e-3, "alpha": 0.5, "beta": 0.5},
                    R={"V": 0.1, "alpha": R_a, "beta": R_a})
    al_kf = np.array([kf.update({"V": V, "alpha_deg": al_raw[i], "beta_deg": 0.0})["alpha_deg"]
                      for i in range(N)])

    t = np.arange(N) / fs
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, al_raw, color=RED, lw=0.5, alpha=0.6, label=f"surowa (σ={al_raw.std():.1f}°)")
    ax.plot(t, al_kf, color=GREEN, lw=2.0, label="po filtrze Kalmana")
    ax.plot(t, alpha_gt, color="k", lw=1.6, ls="--", label="wartość zadana (robot)")
    ax.set_xlabel("czas  [s]"); ax.set_ylabel("kąt natarcia  α  [°]")
    ax.set_title("Estymacja kąta natarcia na profilu schodkowym, V=3 m/s\n"
                 "(filtracja czasowa redukuje szum biały; widoczny lag na krawędziach)")
    ax.legend(loc="upper right", ncol=3)
    _save(fig, "raw_vs_filtered.png")


# ── P7: rozbicie małe vs duże kąty ────────────────────────────────────────────

def plot_angle_breakdown(r):
    ab = r["angle_breakdown"]
    sp = np.array(ab["speed"]); x = np.arange(len(sp)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x - w / 2, ab["small"], w, color=GREEN, label="małe kąty |α|≤10°")
    ax.bar(x + w / 2, ab["large"], w, color=AMBER, label="duże kąty |α|≥20°")
    for i, (s, l) in enumerate(zip(ab["small"], ab["large"])):
        ax.text(i - w / 2, s, f"{s:.1f}°", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, l, f"{l:.1f}°", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{V} m/s" for V in sp])
    ax.set_ylabel("RMSE(α)  [°]")
    ax.set_title("Rozbicie błędu estymacji α: małe vs duże kąty\n"
                 "(przy dużych kątach dochodzi błąd nasycenia/nieliniowości)")
    ax.legend()
    _save(fig, "angle_breakdown.png")


def main():
    print("plotting.py — generowanie figur do SKNTI/img/")
    r = _load()
    plot_alg_diagram()
    plot_rmse_vs_speed(r)
    plot_rmse_all_channels(r)
    plot_degradation_panel(r)
    plot_common_mode(r)
    plot_error_maps(r)
    plot_raw_vs_filtered()
    plot_angle_breakdown(r)
    print("Gotowe.")


if __name__ == "__main__":
    main()
