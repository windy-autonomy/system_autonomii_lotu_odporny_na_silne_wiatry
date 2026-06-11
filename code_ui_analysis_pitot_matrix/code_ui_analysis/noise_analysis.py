"""
Noise characterization from WindShape wind tunnel MAVROS recordings.
Auto-discovers all bags in DATA_DIR that contain /mavros/vfr_hud.
Signal: /mavros/vfr_hud airspeed [m/s], ~10 Hz.
"""

import json, sqlite3, struct, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.stats import shapiro, probplot, skew, kurtosis, norm

RHO      = 1.225   # kg/m³
DATA_DIR = "/home/maciej/Github/windwall_shit/windwall"
OUT_DIR  = "/home/maciej/Github/windwall_shit"

# Known setpoints and clip times.  Bags not listed here are analysed without
# a declared setpoint (v_set=None) and excluded from the model fit.
# t_max clips transients at the END of a recording (seconds from start).
BAG_META = {
    "3ms_mavros_20250909_130645":    {"label": "3ms",           "v_set": 3.0,  "t_max": None},
    "6ms_mavros_20250909_131341":    {"label": "6ms",           "v_set": 6.0,  "t_max": None},
    # The folder named 7_5ms is spin-up data; real 7.5 m/s data is in 133347.
    "7_5ms_mavros_20250909_133106":  {"label": "7.5ms_spinup",  "v_set": None, "t_max": None},
    "mavros_20250909_133347":        {"label": "7.5ms",         "v_set": 7.5,  "t_max": 350.0},
    "gogby_mavros_20250909_134531":  {"label": "gogby",         "v_set": None, "t_max": None},
    "mavros_20250909_121937":        {"label": "121937",        "v_set": None, "t_max": None},
    "mavros_20250909_125318":        {"label": "125318",        "v_set": None, "t_max": None},
    "mavros_20250909_132007":        {"label": "132007",        "v_set": None, "t_max": None},
    "pierwszy_mavros_20250909_130415": {"label": "pierwszy",   "v_set": None, "t_max": None},
}

# Colour palette — setpoint bags get fixed colours, rest get auto colours.
SET_COLORS = {3.0: "#2196F3", 6.0: "#E91E63", 7.5: "#4CAF50"}
UNLABELED_PALETTE = [
    "#FF9800", "#9C27B0", "#795548", "#607D8B", "#00BCD4", "#8BC34A",
]

# ── loaders ───────────────────────────────────────────────────────────────────

def _decode_vfrhud(blob):
    b = bytes(blob)
    offset = 4
    sec, nanosec = struct.unpack_from("<II", b, offset); offset += 8
    fid_len = struct.unpack_from("<I", b, offset)[0]; offset += 4 + fid_len
    if offset % 4: offset += 4 - (offset % 4)
    airspeed = struct.unpack_from("<f", b, offset)[0]
    return sec + nanosec * 1e-9, airspeed


def load_bag(rec_dir):
    db_files = glob.glob(f"{rec_dir}/*.db3")
    if not db_files:
        return None, None
    try:
        conn = sqlite3.connect(db_files[0])
        row = conn.execute("SELECT id FROM topics WHERE name='/mavros/vfr_hud'").fetchone()
        if row is None:
            conn.close(); return None, None
        rows = conn.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (row[0],),
        ).fetchall()
        conn.close()
        if not rows: return None, None
        ts, as_ = zip(*[_decode_vfrhud(r[1]) for r in rows])
        ts, as_ = np.array(ts), np.array(as_)
        idx = np.argsort(ts)
        return ts[idx], as_[idx]
    except Exception:
        return None, None


def load_all():
    """Return list of (rec_name, ts, airspeed, meta) sorted by wall time."""
    out = []
    for rec in os.listdir(DATA_DIR):
        ts, as_ = load_bag(f"{DATA_DIR}/{rec}")
        if ts is None:
            continue
        meta = BAG_META.get(rec, {"label": rec.replace("mavros_20250909_", ""), "v_set": None, "t_max": None})
        out.append((rec, ts, as_, meta))
    out.sort(key=lambda x: x[1][0])
    return out


# ── helpers ───────────────────────────────────────────────────────────────────

def v_to_dp(v):
    return np.sign(v) * 0.5 * RHO * v ** 2


def steady_windows(ts, airspeed, win_sec=20, max_std=0.8, t_max=None):
    t_rel = ts - ts[0]
    end   = t_max if t_max is not None else t_rel[-1]
    mask  = np.zeros(len(ts), dtype=bool)
    step  = win_sec // 2
    for start in np.arange(0, end - win_sec, step):
        m = (t_rel >= start) & (t_rel < start + win_sec)
        if m.sum() < 5: continue
        if airspeed[m].std() < max_std:
            mask |= m
    return mask


def welch_psd(sig, fs):
    nperseg = min(len(sig) // 4, 256)
    f, pxx = signal.welch(sig, fs=fs, nperseg=nperseg, scaling="density")
    return f[f > 0], pxx[f > 0]


def acf_series(sig, max_lag=40):
    sig = sig - sig.mean()
    n   = len(sig)
    c0  = np.dot(sig, sig) / n
    if c0 == 0: return np.arange(max_lag + 1), np.zeros(max_lag + 1)
    lags = np.arange(0, min(max_lag + 1, n))
    vals = np.array([np.dot(sig[:n-k], sig[k:]) / (n * c0) for k in lags])
    return lags, vals


def quantization_lsb(sig):
    vals  = np.sort(np.unique(np.round(sig, 6)))
    if len(vals) < 2: return np.nan
    return float(np.median(np.diff(vals)))


# ── per-bag analysis ──────────────────────────────────────────────────────────

def analyze(label, v_set, ts, airspeed, t_max=None):
    t_rel  = ts - ts[0]
    fs     = 1.0 / np.diff(t_rel).mean()
    dt_std = np.diff(t_rel).std()

    dp     = v_to_dp(airspeed)
    steady = steady_windows(ts, airspeed, t_max=t_max)
    dp_s   = dp[steady]
    v_s    = airspeed[steady]

    if len(dp_s) < 10:
        print(f"  WARNING: {label} has <10 steady samples — skipping statistics")
        return None

    dp_dc    = dp_s.mean()
    dp_resid = dp_s - dp_dc

    n_unique = len(np.unique(np.round(airspeed, 6)))
    lsb_v    = quantization_lsb(airspeed)
    lsb_dp   = RHO * abs(v_s.mean()) * lsb_v if not np.isnan(lsb_v) else np.nan

    std_dp = float(dp_s.std())
    std_v  = float(v_s.std())
    sk     = float(skew(dp_resid))
    kurt_  = float(kurtosis(dp_resid, fisher=True))

    n_steady = int(steady.sum())
    n_sw     = min(n_steady, 5000)
    sw_stat, sw_p = shapiro(dp_resid[:n_sw])
    n_out    = int((np.abs(dp_resid) > 5 * std_dp).sum())

    f_psd, pxx = welch_psd(dp_s, fs)
    floor_idx  = f_psd >= 0.8 * f_psd.max()
    noise_floor_psd = float(np.median(pxx[floor_idx])) if floor_idx.sum() > 0 else np.nan
    sigma_from_floor = float(np.sqrt(noise_floor_psd * (fs / 2))) if not np.isnan(noise_floor_psd) else np.nan

    lags, acf_vals = acf_series(dp_s, max_lag=40)

    low_f = (f_psd > 0.05) & (f_psd < 1.0)
    slope_1f = float(np.polyfit(np.log10(f_psd[low_f]), np.log10(pxx[low_f]), 1)[0]) if low_f.sum() > 3 else np.nan

    q_set = 0.5 * RHO * v_set ** 2 if v_set is not None else np.nan

    return {
        "label":             label,
        "v_set":             v_set,
        "q_set":             q_set,
        "fs":                float(fs),
        "dt_jitter_ms":      float(dt_std * 1000),
        "n_total":           len(ts),
        "n_steady":          n_steady,
        "n_unique_v":        n_unique,
        "lsb_v_ms":          lsb_v,
        "lsb_dp_pa":         lsb_dp,
        "mean_v_measured":   float(v_s.mean()),
        "mean_dp_pa":        float(dp_dc),
        "std_dp_pa":         std_dp,
        "std_v_ms":          std_v,
        "skewness":          sk,
        "excess_kurtosis":   kurt_,
        "sw_stat":           float(sw_stat),
        "sw_p":              float(sw_p),
        "n_outliers_5sig":   n_out,
        "noise_floor_psd":   noise_floor_psd,
        "sigma_from_floor":  sigma_from_floor,
        "slope_1f":          slope_1f,
        "_t_rel":   t_rel,
        "_airspeed": airspeed,
        "_dp":       dp,
        "_dp_s":     dp_s,
        "_dp_resid": dp_resid,
        "_f_psd":    f_psd,
        "_pxx":      pxx,
        "_lags":     lags,
        "_acf":      acf_vals,
        "_steady":   steady,
        "_v_s":      v_s,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def _color(r, unlabeled_idx):
    if r["v_set"] is not None:
        return SET_COLORS.get(r["v_set"], "#607D8B")
    return UNLABELED_PALETTE[unlabeled_idx % len(UNLABELED_PALETTE)]


def _shade_steady(ax, t, steady):
    in_s  = np.where(np.diff(steady.astype(int)) ==  1)[0] + 1
    out_s = np.where(np.diff(steady.astype(int)) == -1)[0] + 1
    if steady[0]:  in_s  = np.r_[0, in_s]
    if steady[-1]: out_s = np.r_[out_s, len(t) - 1]
    for s, e in zip(in_s, out_s):
        ax.axvspan(t[s], t[e], alpha=0.12, color="green")


def plot_per_bag(results):
    """
    One row per bag: [time series | histogram | QQ-plot | ACF].
    Saves noise_analysis.png.
    """
    n   = len(results)
    fig = plt.figure(figsize=(22, 4.5 * n))
    gs  = gridspec.GridSpec(n, 4, figure=fig, hspace=0.55, wspace=0.38)

    ui = 0
    for row, r in enumerate(results):
        c = _color(r, ui)
        if r["v_set"] is None: ui += 1

        t   = r["_t_rel"]
        as_ = r["_airspeed"]

        # ── time series ──────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(t, as_, lw=0.4, color=c, alpha=0.75)
        _shade_steady(ax, t, r["_steady"])
        ax.axhline(r["mean_v_measured"], color="k", ls="--", lw=1,
                   label=f"mean={r['mean_v_measured']:.2f} m/s")
        if r["v_set"] is not None:
            ax.axhline(r["v_set"], color="red", ls=":", lw=1,
                       label=f"setpoint {r['v_set']} m/s")
        ax.set_ylim(-5, 9)
        ax.set_xlabel("Time [s]", fontsize=7)
        ax.set_ylabel("m/s", fontsize=7)
        v_str = f"  set={r['v_set']} m/s" if r["v_set"] else ""
        ax.set_title(
            f"{r['label']}{v_str}\n"
            f"N={r['n_total']}  dur={r['n_total']/r['fs']:.0f}s  "
            f"fs={r['fs']:.1f}Hz  N_steady={r['n_steady']}",
            fontsize=8,
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

        # ── histogram ────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 1])
        dp = r["_dp_resid"]
        bins = min(80, max(10, len(np.unique(np.round(dp, 3)))))
        ax.hist(dp, bins=bins, density=True, color=c, alpha=0.65)
        x = np.linspace(dp.min(), dp.max(), 300)
        ax.plot(x, norm.pdf(x, 0, dp.std()), "k--", lw=1.2, label="Gaussian")
        ax.set_xlabel("ΔP residual [Pa]", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.set_title(
            f"Histogram\nσ={r['std_dp_pa']:.3f} Pa  "
            f"kurt={r['excess_kurtosis']:.2f}  skew={r['skewness']:.2f}",
            fontsize=8,
        )
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # ── QQ-plot ──────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 2])
        (osm, osr), (slope, intercept, _) = probplot(dp, dist="norm", fit=True)
        ax.plot(osm, osr, ".", ms=1.5, color=c, alpha=0.5)
        lx = np.array([osm[0], osm[-1]])
        ax.plot(lx, slope * lx + intercept, "k--", lw=1.2)
        ax.set_xlabel("Theoretical quantiles", fontsize=7)
        ax.set_ylabel("Sample [Pa]", fontsize=7)
        ax.set_title(f"QQ-plot\nSW p={r['sw_p']:.1e}", fontsize=8)
        ax.grid(alpha=0.3)

        # ── ACF ──────────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 3])
        ax.bar(r["_lags"], r["_acf"], color=c, alpha=0.75, width=0.8)
        ci = 1.96 / np.sqrt(r["n_steady"])
        ax.axhline( ci, color="k", ls="--", lw=0.9)
        ax.axhline(-ci, color="k", ls="--", lw=0.9)
        ax.axhline(0,   color="k", lw=0.4)
        ax.set_xlabel("Lag [samples]", fontsize=7)
        ax.set_ylabel("ACF", fontsize=7)
        ax.set_title(
            f"ACF of ΔP\n1/f slope={r['slope_1f']:.2f}",
            fontsize=8,
        )
        ax.set_ylim([-0.4, 1.05])
        ax.grid(alpha=0.3)

    plt.suptitle(
        "WindShape tunnel — noise characterization of ALL bags\n"
        "(green shading = stationary windows used for statistics)",
        fontsize=12, y=1.001,
    )
    out = f"{OUT_DIR}/noise_analysis.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def plot_psd_comparison(results):
    """All PSDs on one log-log plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ui = 0
    for r in results:
        c = _color(r, ui)
        if r["v_set"] is None: ui += 1
        lbl = f"{r['label']}  σ_ΔP={r['std_dp_pa']:.3f} Pa"
        if r["v_set"]: lbl += f"  (set {r['v_set']} m/s)"
        ax.loglog(r["_f_psd"], r["_pxx"], lw=1.4, color=c, alpha=0.85, label=lbl)
        ax.axhline(r["noise_floor_psd"], ls=":", color=c, alpha=0.5, lw=1)

    # reference slopes
    f_ref = np.array([0.05, 5.0])
    ax.loglog(f_ref, 0.05 * f_ref**-1, "gray", ls="--", lw=0.9, alpha=0.5, label="−1 slope (1/f)")
    ax.loglog(f_ref, 0.005 * f_ref**-2, "gray", ls="-.", lw=0.9, alpha=0.5, label="−2 slope (1/f²)")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [(Pa)²/Hz]")
    ax.set_title("Welch PSD of ΔP — steady-state windows  (dotted = high-f floor)")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim([0.02, 5])
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out = f"{OUT_DIR}/noise_psd.png"
    plt.savefig(out, dpi=140)
    print(f"Saved {out}")
    plt.close()


def plot_sigma_vs_v(results):
    """σ_ΔP vs setpoint velocity — model fit on bags with known setpoints."""
    setpoint_res = [r for r in results if r["v_set"] is not None]
    if len(setpoint_res) < 2:
        print("Not enough setpoint bags for model fit.")
        return np.nan, np.nan

    vs  = np.array([r["v_set"]     for r in setpoint_res])
    ss  = np.array([r["std_dp_pa"] for r in setpoint_res])
    qs  = np.array([r["q_set"]     for r in setpoint_res])

    A = np.column_stack([np.ones(len(qs)), qs**2])
    coeffs, _, _, _ = np.linalg.lstsq(A, ss**2, rcond=None)
    sigma_e2, k2 = coeffs
    sigma_e2 = max(float(sigma_e2), 0.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ui = 0
    for r in results:
        c = _color(r, ui)
        if r["v_set"] is None:
            ui += 1
            ax.plot(r["mean_v_measured"], r["std_dp_pa"], "x", ms=8, color=c,
                    label=f"{r['label']} (unlabeled, mean V={r['mean_v_measured']:.1f})")
        else:
            ax.plot(r["v_set"], r["std_dp_pa"], "o", ms=12, color=c,
                    label=f"{r['label']}  σ_ΔP={r['std_dp_pa']:.3f} Pa")

    v_plot = np.linspace(0, 9, 300)
    q_plot = 0.5 * RHO * v_plot**2
    sigma_model = np.sqrt(np.maximum(sigma_e2 + k2 * q_plot**2, 0))
    ax.plot(v_plot, sigma_model, "k--", lw=1.8,
            label=f"Model σ²=σ_e²+(k·q)²\nσ_e={np.sqrt(sigma_e2):.3f} Pa  k={np.sqrt(abs(k2)):.4f}")

    ax.set_xlabel("V [m/s]  (setpoint for labeled bags, mean measured for unlabeled)")
    ax.set_ylabel("σ_ΔP [Pa]  (steady-state std)")
    ax.set_title("σ_ΔP vs wind speed — model fit (circles = setpoint bags)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = f"{OUT_DIR}/sigma_vs_v.png"
    plt.savefig(out, dpi=140)
    print(f"Saved {out}")
    plt.close()

    return float(np.sqrt(sigma_e2)), float(np.sqrt(abs(k2)))


# ── report ────────────────────────────────────────────────────────────────────

def print_report(results, sigma_e, k):
    print("\n" + "=" * 72)
    print("WINDWALL AIRSPEED SENSOR NOISE CHARACTERIZATION — ALL BAGS")
    print("=" * 72)
    print("\n⚠  BANDWIDTH LIMITATION")
    print("   fs ≈ 10 Hz → Nyquist = 5 Hz.  Sensor electronic white noise is")
    print("   typically >20 Hz and cannot be resolved here.  σ_ΔP values are")
    print("   UPPER BOUNDS (turbulence + aliased sensor noise combined).")

    # Summary table
    print(f"\n{'─'*72}")
    hdr = f"  {'Label':<16} {'V_set':>6} {'V_meas':>7} {'N_tot':>6} {'N_stead':>7} " \
          f"{'σ_ΔP(Pa)':>9} {'SW_p':>9} {'1/f':>6}"
    print(hdr)
    print(f"  {'─'*68}")
    for r in results:
        vset = f"{r['v_set']:.1f}" if r["v_set"] else "  — "
        print(
            f"  {r['label']:<16} {vset:>6} {r['mean_v_measured']:>7.3f} "
            f"{r['n_total']:>6} {r['n_steady']:>7} "
            f"{r['std_dp_pa']:>9.4f} {r['sw_p']:>9.2e} {r['slope_1f']:>6.2f}"
        )

    # Per-bag details
    for r in results:
        vset_str = f"{r['v_set']} m/s" if r["v_set"] else "unknown"
        print(f"\n{'─'*60}")
        print(f"  {r['label']}  (declared setpoint: {vset_str})")
        print(f"{'─'*60}")
        print(f"  Duration        : {r['n_total']/r['fs']:.0f} s  ({r['n_total']} samples)")
        print(f"  fs              : {r['fs']:.2f} Hz   jitter ±{r['dt_jitter_ms']:.1f} ms")
        print(f"  Unique V vals   : {r['n_unique_v']}   LSB_V ≈ {r['lsb_v_ms']:.4f} m/s   LSB_ΔP ≈ {r['lsb_dp_pa']:.3f} Pa")
        print(f"  Mean V measured : {r['mean_v_measured']:.3f} m/s")
        print(f"  Mean ΔP         : {r['mean_dp_pa']:.3f} Pa")
        print(f"\n  Steady windows  : {r['n_steady']} samples")
        print(f"  σ_ΔP            : {r['std_dp_pa']:.4f} Pa")
        print(f"  σ_V measured    : {r['std_v_ms']:.4f} m/s")
        if r["v_set"]:
            tu = r["std_v_ms"] / r["v_set"] * 100
            print(f"  TI = σ_V/V_set  : {tu:.1f}%  (upper bound)")
        print(f"  Skewness        : {r['skewness']:.3f}")
        print(f"  Excess kurtosis : {r['excess_kurtosis']:.3f}")
        sw_tag = "(non-Gaussian)" if r["sw_p"] < 0.05 else "(Gaussian compatible)"
        print(f"  Shapiro-Wilk p  : {r['sw_p']:.2e}  {sw_tag}")
        print(f"  Outliers >5σ    : {r['n_outliers_5sig']} ({100*r['n_outliers_5sig']/r['n_steady']:.2f}%)")
        print(f"  High-f PSD floor: {r['noise_floor_psd']:.4e} (Pa)²/Hz")
        print(f"  σ from floor×BW : {r['sigma_from_floor']:.4f} Pa")
        print(f"  1/f slope       : {r['slope_1f']:.2f}  (0=white, −1=1/f, −2=1/f²)")

    print(f"\n{'='*72}")
    print("  NOISE MODEL   σ_ΔP²(V) = σ_e² + (k·q(V))²   [setpoint bags only]")
    print(f"{'='*72}")
    print(f"  σ_e  (zero-speed floor)  = {sigma_e:.4f} Pa")
    print(f"  k    (pressure fraction) = {k:.5f}")
    print()
    print("  NOTE: fit is dominated by turbulence, not sensor electronics.")
    print("  To isolate σ_sensor: re-record at ≥100 Hz on a raw ADC topic.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update({"font.size": 9})

    bags = load_all()
    print(f"Found {len(bags)} bags with /mavros/vfr_hud\n")

    results = []
    for rec, ts, as_, meta in bags:
        print(f"  Analysing {meta['label']} ...")
        r = analyze(meta["label"], meta["v_set"], ts, as_, t_max=meta["t_max"])
        if r is not None:
            results.append(r)

    print(f"\nPlotting {len(results)} results...")
    plot_per_bag(results)
    plot_psd_comparison(results)
    sigma_e, k = plot_sigma_vs_v(results)
    print_report(results, sigma_e, k)


if __name__ == "__main__":
    main()
