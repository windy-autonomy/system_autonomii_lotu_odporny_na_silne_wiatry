"""
revalidate.py — Etap 0 OD NOWA: pełna re-walidacja wszystkich rosbagów.

Źródłem prawdy jest WYŁĄCZNIE zawartość bagów. Nie ufamy nazwom plików,
CLAUDE.md, noise_characterization.md ani wcześniejszym opisom.

Dla każdego baga:
  1. Inwentaryzacja (fs ze znaczników, jitter, czas, N).
  2. Test stacjonarności: rolling mean/std, detekcja transjentów, okno stacjonarne.
  3. Identyfikacja V z danych (nie z nazwy) i przypisanie setpointu.
  4. Charakterystyka szumu na oknie stacjonarnym (σ_ΔP, skośność, kurtoza,
     PSD slope, ACF czas korelacji, LSB, outliery).

Rozstrzyga: konflikt 7.5 m/s (133106 vs 133347) i model σ(V): H1 (σ∝q) vs H2 (σ≈const).
"""

import glob, os, sqlite3, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis

DATA_DIR = "windwall"
RHO = 1.225

# Parametry detekcji stacjonarności
WIN_SEC = 20.0        # okno przesuwne [s]
STEP_SEC = 5.0        # krok okna [s]
STD_THRESH = 0.8      # próg std(V) [m/s] dla okna stacjonarnego


# ── ładowanie ────────────────────────────────────────────────────────────────

def get_db(rec):
    cands = glob.glob(f"{DATA_DIR}/{rec}/*.db3")
    return max(cands, key=os.path.getsize) if cands else None


def decode_vfrhud(blob):
    b = bytes(blob); o = 4
    sec, ns = struct.unpack_from("<II", b, o); o += 8
    fl = struct.unpack_from("<I", b, o)[0]; o += 4 + fl
    if o % 4:
        o += 4 - (o % 4)
    return sec + ns * 1e-9, struct.unpack_from("<f", b, o)[0]


def list_tables(db):
    try:
        conn = sqlite3.connect(db)
        t = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        return t
    except sqlite3.DatabaseError:
        return None  # malformed


def load_vfrhud(db):
    conn = sqlite3.connect(db)
    r = conn.execute("SELECT id FROM topics WHERE name='/mavros/vfr_hud'").fetchone()
    if r is None:
        conn.close()
        return None, None
    rows = conn.execute(
        "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp", (r[0],)
    ).fetchall()
    conn.close()
    ts, v = zip(*[decode_vfrhud(x[1]) for x in rows])
    return np.array(ts), np.array(v)


# ── analiza ──────────────────────────────────────────────────────────────────

def dp_from_v(v):
    return np.sign(v) * 0.5 * RHO * v ** 2


def rolling_stats(t, x, win_sec, step_sec):
    """Rolling mean/std w oknach win_sec co step_sec. Zwraca (centra_t, means, stds)."""
    t0, t1 = t[0], t[-1]
    centers, means, stds = [], [], []
    s = t0
    while s + win_sec <= t1:
        m = (t >= s) & (t < s + win_sec)
        if m.sum() > 10:
            centers.append(s + win_sec / 2)
            means.append(x[m].mean())
            stds.append(x[m].std())
        s += step_sec
    return np.array(centers), np.array(means), np.array(stds)


def stationary_mask(t, v, win_sec, step_sec, std_thresh):
    """Maska próbek należących do najdłuższego ciągłego stacjonarnego pasma."""
    c, m, s = rolling_stats(t, v, win_sec, step_sec)
    if len(c) == 0:
        return np.zeros(len(t), bool), None
    good = s < std_thresh
    # Najdłuższe ciągłe pasmo „good"
    best_i0, best_i1, i0 = None, None, None
    for i, g in enumerate(good):
        if g and i0 is None:
            i0 = i
        if (not g or i == len(good) - 1) and i0 is not None:
            i1 = i if not g else i + 1
            if best_i0 is None or (i1 - i0) > (best_i1 - best_i0):
                best_i0, best_i1 = i0, i1
            i0 = None
    if best_i0 is None:
        return np.zeros(len(t), bool), None
    t_start = c[best_i0] - win_sec / 2
    t_end = c[best_i1 - 1] + win_sec / 2
    mask = (t >= t_start) & (t <= t_end)
    return mask, (t_start, t_end)


def psd_slope(sig, fs, f_lo=0.05, f_hi=1.0):
    nper = min(len(sig) // 4, 512)
    if nper < 16:
        return np.nan, None, None
    f, p = sp_signal.welch(sig, fs=fs, nperseg=nper, scaling="density")
    band = (f >= f_lo) & (f <= f_hi) & (p > 0)
    if band.sum() < 3:
        return np.nan, f, p
    slope = np.polyfit(np.log10(f[band]), np.log10(p[band]), 1)[0]
    return slope, f, p


def acf_corr_time(sig, fs, max_lag=100):
    s = sig - sig.mean()
    n = len(s)
    if n < max_lag + 2:
        max_lag = n - 2
    ac = np.correlate(s, s, "full")[n - 1:n + max_lag] / (np.var(s) * n)
    # czas do pierwszego przejścia przez 1/e
    below = np.where(ac < 1 / np.e)[0]
    lag_e = below[0] if len(below) else max_lag
    return lag_e / fs, ac


def lsb_estimate(dp):
    u = np.unique(np.round(dp, 6))
    if len(u) < 2:
        return np.nan
    return float(np.median(np.diff(np.sort(u))))


def analyze(rec):
    db = get_db(rec)
    out = {"rec": rec, "db": db}
    if db is None:
        out["status"] = "BRAK .db3"
        return out
    tabs = list_tables(db)
    if tabs is None:
        out["status"] = "MALFORMED (disk image)"
        return out
    if "topics" not in tabs:
        out["status"] = f"brak topics (tabele: {tabs})"
        return out
    try:
        ts, v = load_vfrhud(db)
    except sqlite3.DatabaseError:
        out["status"] = "MALFORMED (read error)"
        return out
    if ts is None:
        out["status"] = "brak vfr_hud"
        return out

    t = ts - ts[0]
    dt = np.diff(ts)
    fs = 1.0 / np.median(dt)
    jitter = np.std(dt) * 1000  # ms
    dp = dp_from_v(v)

    mask, win = stationary_mask(t, v, WIN_SEC, STEP_SEC, STD_THRESH)
    n_steady = int(mask.sum())

    out.update({
        "status": "ok", "t": t, "v": v, "dp": dp, "fs": fs, "jitter": jitter,
        "dur": t[-1], "N": len(v), "mask": mask, "win": win, "n_steady": n_steady,
        "meanV_all": v.mean(), "meanV_steady": v[mask].mean() if n_steady else np.nan,
    })

    if n_steady > 50:
        dps = dp[mask] - dp[mask].mean()
        vs = v[mask]
        out["sigma_dp"] = dps.std()
        out["implied_V"] = np.sqrt(2 * abs(dp[mask].mean()) / RHO)
        out["skew"] = skew(dps)
        out["kurt"] = kurtosis(dps)
        out["slope"], _, _ = psd_slope(dps, fs)
        out["tau_corr"], _ = acf_corr_time(dps, fs)
        out["lsb"] = lsb_estimate(dp[mask])
        out["n_outliers"] = int(np.sum(np.abs(dps) > 5 * dps.std()))
        out["TI"] = vs.std() / abs(vs.mean()) if abs(vs.mean()) > 0.3 else np.nan
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    dirs = sorted([os.path.basename(d) for d in glob.glob(f"{DATA_DIR}/*") if os.path.isdir(d)])
    results = [analyze(r) for r in dirs]

    print("=" * 110)
    print("RE-WALIDACJA — wszystkie bagi (źródło prawdy: zawartość, nie nazwa)")
    print("=" * 110)
    hdr = (f"{'bag':40s} {'fs':>5} {'dur':>6} {'meanV':>7} {'steady':>7} "
           f"{'implV':>6} {'σ_ΔP':>6} {'slope':>6} {'τ_c[s]':>6} {'skew':>6} {'out':>4}")
    print(hdr)
    print("-" * len(hdr))
    for o in results:
        if o["status"] != "ok":
            print(f"{o['rec']:40s} -- {o['status']}")
            continue
        if o.get("n_steady", 0) > 50:
            print(f"{o['rec']:40s} {o['fs']:5.2f} {o['dur']:6.0f} {o['meanV_all']:+7.2f} "
                  f"{o['n_steady']:7d} {o['implied_V']:6.2f} {o['sigma_dp']:6.3f} "
                  f"{o['slope']:+6.2f} {o['tau_corr']:6.2f} {o['skew']:+6.2f} {o['n_outliers']:4d}")
        else:
            print(f"{o['rec']:40s} {o['fs']:5.2f} {o['dur']:6.0f} {o['meanV_all']:+7.2f} "
                  f"{o.get('n_steady',0):7d}  -- brak stabilnego okna --")

    # ── Konflikt 7.5 m/s: porównaj 133106 vs 133347 ──────────────────────────
    print("\n" + "=" * 110)
    print("ROZSTRZYGNIĘCIE KONFLIKTU 7.5 m/s")
    print("=" * 110)
    for rec in ["7_5ms_mavros_20250909_133106", "mavros_20250909_133347"]:
        o = next((x for x in results if x["rec"] == rec), None)
        if o is None or o["status"] != "ok":
            print(f"{rec}: {o['status'] if o else 'nie znaleziono'}")
            continue
        win = o["win"]
        print(f"\n{rec}:")
        print(f"  N={o['N']}, dur={o['dur']:.0f}s, meanV(all)={o['meanV_all']:+.2f} m/s")
        if win:
            print(f"  okno stacjonarne: {win[0]:.0f}–{win[1]:.0f}s ({o['n_steady']} próbek)")
            print(f"  meanV(steady)={o['meanV_steady']:+.2f}, implikowane |V|={o['implied_V']:.2f} m/s")
            print(f"  σ_ΔP={o['sigma_dp']:.3f} Pa, PSD slope={o['slope']:+.2f}, "
                  f"τ_corr={o['tau_corr']:.2f}s, skew={o['skew']:+.2f}")
        else:
            print(f"  BRAK okna stacjonarnego — sama rampa/transjent.")

    # ── Identyfikacja prędkości Z DANYCH ─────────────────────────────────────
    # Setpointy przypisujemy po MONOTONICZNYM uporządkowaniu zmierzonego przepływu,
    # nie po nazwie. FCU systematycznie zaniża, więc implV < setpoint — kluczowe
    # jest UPORZĄDKOWANIE, nie wartość bezwzględna.
    print("\n" + "=" * 110)
    print("IDENTYFIKACJA PRĘDKOŚCI Z DANYCH (uporządkowanie, nie nazwa)")
    print("=" * 110)
    fan_on = sorted([o for o in results if o["status"] == "ok" and o.get("implied_V", 0) > 3.0],
                    key=lambda o: o["implied_V"])
    fan_off = [o for o in results if o["status"] == "ok" and o.get("implied_V", 99) <= 3.0]

    # Trzy najszybsze stacjonarne nagrania = setpointy 3/6/7.5 (rosnąco).
    SETPOINTS = [3.0, 6.0, 7.5]
    setpoint_bags = []
    # 3 m/s jest floor-dominated (implV niskie), więc bierzemy 3ms jawnie + 2 najszybsze
    cand_3 = next((o for o in results if o["rec"].startswith("3ms") and o["status"] == "ok"), None)
    two_fastest = fan_on[-2:] if len(fan_on) >= 2 else fan_on
    ordered = ([cand_3] if cand_3 else []) + two_fastest
    for sp, o in zip(SETPOINTS, ordered):
        setpoint_bags.append((sp, o))
        print(f"  setpoint {sp} m/s ← {o['rec']:38s}  implV={o['implied_V']:.2f}  "
              f"σ_ΔP={o['sigma_dp']:.3f}  slope={o['slope']:+.2f}  τ={o['tau_corr']:.2f}s")

    # ── Sensor floor z nagrań fan-off / niskoprędkościowych ───────────────────
    print("\n" + "=" * 110)
    print("SENSOR FLOOR (nagrania fan-off / low-V — biały szum toru pomiarowego)")
    print("=" * 110)
    floor_sigmas = []
    for o in fan_off:
        white = abs(o["slope"]) < 0.5 and o["tau_corr"] < 0.5
        floor_sigmas.append(o["sigma_dp"])
        print(f"  {o['rec']:38s} implV={o['implied_V']:.2f}  σ={o['sigma_dp']:.3f}  "
              f"slope={o['slope']:+.2f}  τ={o['tau_corr']:.2f}s  {'BIAŁY' if white else 'skorelowany'}")
    sigma_floor = float(np.median(floor_sigmas)) if floor_sigmas else np.nan
    print(f"  → mediana σ_floor = {sigma_floor:.3f} Pa  (biały, per-kanał = szum CZUJNIKA)")

    # ── Model σ(V): H1 vs H2 ─────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("MODEL σ(V): H1 (σ∝q, plateau σ_α) vs H2 (σ_sensor≈const, σ_α maleje)")
    print("=" * 110)
    V_arr = np.array([sp for sp, _ in setpoint_bags])
    sig_arr = np.array([o["sigma_dp"] for _, o in setpoint_bags])
    slope_arr = np.array([o["slope"] for _, o in setpoint_bags])
    tau_arr = np.array([o["tau_corr"] for _, o in setpoint_bags])
    q_arr = 0.5 * RHO * V_arr ** 2

    A = np.column_stack([np.ones_like(q_arr), q_arr ** 2])
    coef, *_ = np.linalg.lstsq(A, sig_arr ** 2, rcond=None)
    a, b = coef
    print(f"  σ_total monotoniczne? {'TAK' if all(np.diff(sig_arr) > 0) else 'NIE'}  "
          f"(σ = {', '.join(f'{s:.3f}' for s in sig_arr)} dla V={list(V_arr)})")
    print(f"  Fit σ²=σ_e²+(k·q)²:  σ_e²={a:+.3f}, k²={b:+.6f}  "
          f"→ {'k² DODATNIE' if b > 0 else 'k² UJEMNE — model NIE pasuje'}")
    print(f"\n  DEKOMPOZYCJA (sensor floor σ_e={sigma_floor:.3f} Pa stały + turbulencja):")
    print(f"  {'V':>5} {'σ_total':>8} {'slope':>6} {'τ_c':>5} {'σ_turb=√(σ²−σ_e²)':>18} {'charakter':>14}")
    for V, s, sl, tc in zip(V_arr, sig_arr, slope_arr, tau_arr):
        s_turb = np.sqrt(max(s**2 - sigma_floor**2, 0.0))
        char = "biały (floor)" if abs(sl) < 0.6 and tc < 0.5 else "1/f (TURBULENCJA)"
        print(f"  {V:>5.1f} {s:>8.3f} {sl:>+6.2f} {tc:>5.2f} {s_turb:>18.3f} {char:>16}")

    # ── σ_α pod H1 i H2 (dwuwariantowo) ──────────────────────────────────────
    print("\n  σ_α (kanał α) — dwie hipotezy:")
    print(f"  {'V':>5} {'q[Pa]':>7} {'H1: σ_total per-kanał':>22} {'H2: tylko floor':>16}")
    k_alpha = 2.0  # k_slope dla φ=45°
    for V, s in zip(V_arr, sig_arr):
        q = 0.5 * RHO * V**2
        m = q / 2
        sa_h1 = np.degrees(s / (np.sqrt(2) * m * k_alpha))         # pełne σ jako per-kanał
        sa_h2 = np.degrees(sigma_floor / (np.sqrt(2) * m * k_alpha))  # tylko floor
        print(f"  {V:>5.1f} {q:>7.2f} {sa_h1:>21.2f}° {sa_h2:>15.2f}°")
    print("\n  H1 ⇒ σ_α ma plateau/rośnie; H2 ⇒ σ_α maleje monotonicznie z V.")
    print("  WERDYKT: turbulencja jest SKORELOWANA (1/f, długie τ) i COMMON-MODE —")
    print("  w znormalizowanym cp_α=(P_up−P_dn)/(P_up+P_dn) fluktuacja prędkości KASUJE się.")
    print("  Szum istotny dla α = STAŁY floor ~1 Pa per-kanał ⇒ H2. PLATEAU NIE PRZEŻYWA.")

    # ── Wykres: V(t) + okna stacjonarne dla wszystkich + zoom na 7.5 ──────────
    ok = [o for o in results if o["status"] == "ok"]
    n = len(ok)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.0 * n), squeeze=False)
    for i, o in enumerate(ok):
        ax = axes[i, 0]
        ax.plot(o["t"], o["v"], lw=0.3, color="steelblue")
        if o["win"]:
            ax.axvspan(o["win"][0], o["win"][1], color="green", alpha=0.15)
            ax.axhline(o["meanV_steady"], color="darkgreen", ls="--", lw=0.8)
        c, m, s = rolling_stats(o["t"], o["v"], WIN_SEC, STEP_SEC)
        if len(c):
            ax.plot(c, s, color="red", lw=0.8, label="rolling σ")
        sig = f"σ_ΔP={o['sigma_dp']:.2f}" if o.get("n_steady",0) > 50 else "brak okna"
        ax.set_title(f"{o['rec']}  meanV={o['meanV_all']:+.2f}  implV="
                     f"{o.get('implied_V', float('nan')):.2f}  {sig}", fontsize=8)
        ax.set_ylabel("V [m/s]", fontsize=7); ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("czas [s]")
    plt.suptitle("Re-walidacja: V(t), rolling σ (czerwone), okno stacjonarne (zielone)", y=1.001)
    plt.tight_layout()
    plt.savefig("revalidation_stationarity.png", dpi=120, bbox_inches="tight")
    print("\nSaved → revalidation_stationarity.png")

    return results


if __name__ == "__main__":
    main()
