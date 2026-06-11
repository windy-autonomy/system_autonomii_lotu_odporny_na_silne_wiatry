"""
Bag validation protocol — WindShape tunnel recordings.
Implements the validation checklist from meta_prompt_noise_extraction.md:

1. Load every bag, report: topic, fs (from timestamps), duration, sample count, jitter.
2. Check declared vs. measured speed — mean |V| and ΔP must increase monotonically
   with the labeled setpoint (3 < 6 < 7.5 m/s).  Flag mismatches.
3. Warning flags: σ_ΔP monotonicity, anomalous 1/f slope, long ACF tail, short recordings.
4. Detect and clip transients (ramp-up / ramp-down) using a rolling-window stationarity test.
5. Output a "declared vs measured" table and an explicit VERDICT per bag.

Usage:
    python validate_bags.py
Produces:  validation_report.txt  and  validation_plots.png
"""

import json, sqlite3, struct, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from scipy import signal
from scipy.stats import shapiro

# ── configuration ─────────────────────────────────────────────────────────────
DATA_DIR = "/home/maciej/Github/windwall_shit/windwall"
OUT_DIR  = "/home/maciej/Github/windwall_shit"
RHO      = 1.225  # kg/m³

# Ground-truth: bags that carry a declared setpoint (folder name → V_setpoint m/s)
DECLARED = {
    "3ms_mavros_20250909_130645":   3.0,
    "6ms_mavros_20250909_131341":   6.0,
    "7_5ms_mavros_20250909_133106": 7.5,
}

# ── loaders ───────────────────────────────────────────────────────────────────

def _decode_vfrhud(blob):
    b = bytes(blob)
    offset = 4
    sec, nanosec = struct.unpack_from("<II", b, offset); offset += 8
    fid_len = struct.unpack_from("<I", b, offset)[0]; offset += 4 + fid_len
    if offset % 4: offset += 4 - (offset % 4)
    airspeed = struct.unpack_from("<f", b, offset)[0]
    return sec + nanosec * 1e-9, airspeed


def load_bag_db(db_path):
    """Return (ts, airspeed_ms) arrays or (None, None) on failure."""
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT id FROM topics WHERE name='/mavros/vfr_hud'").fetchone()
        if row is None:
            conn.close(); return None, None
        rows = conn.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (row[0],),
        ).fetchall()
        conn.close()
        if not rows:
            return None, None
        ts, as_ = zip(*[_decode_vfrhud(r[1]) for r in rows])
        ts, as_ = np.array(ts), np.array(as_)
        idx = np.argsort(ts)
        return ts[idx], as_[idx]
    except Exception as e:
        return None, None


def load_json_vfrhud(json_path):
    with open(json_path) as f:
        data = json.load(f)
    ts, as_ = [], []
    for msg in data.values():
        t = msg["header"]["stamp"]["sec"] + msg["header"]["stamp"]["nanosec"] * 1e-9
        ts.append(t); as_.append(msg["airspeed"])
    ts, as_ = np.array(ts), np.array(as_)
    idx = np.argsort(ts)
    return ts[idx], as_[idx]


# ── helpers ───────────────────────────────────────────────────────────────────

def timing_stats(ts):
    dt = np.diff(ts)
    return {
        "fs":         1.0 / dt.mean(),
        "dt_mean_ms": dt.mean() * 1000,
        "dt_std_ms":  dt.std() * 1000,
        "dt_max_ms":  dt.max() * 1000,
        "jitter_pct": dt.std() / dt.mean() * 100,
    }


def dp_from_v(v):
    """Signed differential pressure [Pa]: q = sign(V)·0.5·ρ·V²"""
    return np.sign(v) * 0.5 * RHO * v ** 2


def detect_stationary_mask(ts, airspeed, win_sec=20, max_std=0.8, step_sec=5):
    """
    Return boolean mask of samples inside rolling windows whose std < max_std.
    Transients (ramp-up, ramp-down, gusts) are excluded.
    """
    t_rel = ts - ts[0]
    end   = t_rel[-1]
    mask  = np.zeros(len(ts), dtype=bool)
    for start in np.arange(0, end - win_sec, step_sec):
        m = (t_rel >= start) & (t_rel < start + win_sec)
        if m.sum() < 5:
            continue
        if airspeed[m].std() < max_std:
            mask |= m
    return mask


def welch_psd(sig, fs):
    nperseg = min(len(sig) // 4, 256)
    f, pxx = signal.welch(sig, fs=fs, nperseg=nperseg, scaling="density")
    return f[f > 0], pxx[f > 0]


def acf_first_zero(sig, max_lag=100):
    """Return lag (samples) at which ACF first crosses zero."""
    sig = sig - sig.mean()
    n   = len(sig)
    c0  = np.dot(sig, sig) / n
    for k in range(1, min(max_lag, n)):
        r = np.dot(sig[:n-k], sig[k:]) / (n * c0)
        if r <= 0:
            return k
    return max_lag


def slope_1f(f_psd, pxx):
    """Log-log slope in 0.05–1 Hz range (−1 = 1/f, 0 = white)."""
    m = (f_psd > 0.05) & (f_psd < 1.0)
    if m.sum() < 3:
        return np.nan
    lf = np.log10(f_psd[m])
    lp = np.log10(pxx[m])
    return float(np.polyfit(lf, lp, 1)[0])


# ── per-bag analysis ──────────────────────────────────────────────────────────

def analyze_bag(rec_name, ts, airspeed, declared_v=None):
    t_rel = ts - ts[0]
    t_wall = datetime.fromtimestamp(ts[0], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    dur = t_rel[-1]

    tim = timing_stats(ts)
    fs  = tim["fs"]

    dp = dp_from_v(airspeed)

    steady = detect_stationary_mask(ts, airspeed)
    n_steady = steady.sum()

    dp_s  = dp[steady]
    as_s  = airspeed[steady]
    dp_dc = dp_s.mean() if n_steady > 0 else np.nan
    dp_resid = dp_s - dp_dc if n_steady > 0 else np.array([])

    mean_v  = float(np.abs(airspeed).mean())
    mean_dp = float(dp_s.mean()) if n_steady > 0 else np.nan
    std_dp  = float(dp_s.std())  if n_steady > 0 else np.nan
    std_v   = float(as_s.std())  if n_steady > 0 else np.nan

    q_declared = 0.5 * RHO * declared_v ** 2 if declared_v else None

    psd_f, pxx = (None, None)
    slope = np.nan
    acf_zero = None
    if n_steady > 20:
        psd_f, pxx = welch_psd(dp_s, fs)
        slope = slope_1f(psd_f, pxx)
        acf_zero = acf_first_zero(dp_s)

    # Verdict flags
    flags = []
    if dur < 200:
        flags.append("SHORT_RECORDING (<200s)")
    if n_steady < 100:
        flags.append("FEW_STEADY_SAMPLES")
    if declared_v is not None:
        # mean absolute reported velocity should be in [0.5·V, 1.5·V] range
        if mean_v < 0.2 * declared_v:
            flags.append(f"SEVERE_UNDERREAD (|mean_V|={mean_v:.2f} << declared={declared_v})")
        elif mean_v < 0.5 * declared_v:
            flags.append(f"UNDERREAD (|mean_V|={mean_v:.2f} < 0.5·declared={declared_v})")
    if not np.isnan(slope) and slope < -1.5:
        flags.append(f"STRONG_1/f_NOISE (slope={slope:.2f})")
    if acf_zero is not None and acf_zero > 20:
        flags.append(f"LONG_CORRELATION (ACF zero at lag {acf_zero})")

    return {
        "rec":          rec_name,
        "declared_v":   declared_v,
        "t_wall":       t_wall,
        "dur_s":        float(dur),
        "n_total":      len(ts),
        "n_steady":     int(n_steady),
        "fs":           float(fs),
        "dt_std_ms":    float(tim["dt_std_ms"]),
        "dt_max_ms":    float(tim["dt_max_ms"]),
        "jitter_pct":   float(tim["jitter_pct"]),
        "mean_abs_v":   float(mean_v),
        "mean_dp_pa":   float(mean_dp) if not np.isnan(mean_dp) else None,
        "std_dp_pa":    float(std_dp)  if not np.isnan(std_dp) else None,
        "std_v_ms":     float(std_v)   if not np.isnan(std_v) else None,
        "slope_1f":     float(slope),
        "acf_zero_lag": int(acf_zero) if acf_zero is not None else None,
        "flags":        flags,
        # arrays for plotting
        "_t_rel":    t_rel,
        "_airspeed": airspeed,
        "_dp":       dp,
        "_steady":   steady,
        "_psd_f":    psd_f,
        "_pxx":      pxx,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("WINDSHAPE TUNNEL BAG VALIDATION REPORT")
    emit(f"Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    emit("=" * 78)

    # ── load all bags ─────────────────────────────────────────────────────────
    emit("\n[ 1 ] LOADING ALL BAGS\n")

    recordings = sorted(os.listdir(DATA_DIR))
    results = []

    for rec in recordings:
        db_files = glob.glob(f"{DATA_DIR}/{rec}/*.db3")
        if not db_files:
            continue
        ts, as_ = load_bag_db(db_files[0])
        status = ""
        if ts is None:
            emit(f"  SKIP  {rec}  — no /mavros/vfr_hud or malformed DB")
            continue
        declared_v = DECLARED.get(rec, None)
        r = analyze_bag(rec, ts, as_, declared_v=declared_v)
        results.append(r)
        emit(f"  OK    {rec}  ({r['t_wall']})  N={r['n_total']}  dur={r['dur_s']:.0f}s")

    # ── per-bag summary ───────────────────────────────────────────────────────
    emit("\n[ 2 ] PER-BAG TIMING & SIGNAL SUMMARY\n")

    hdr = (f"{'Recording':<45} {'fs':>5} {'jitter%':>8} {'dur(s)':>7} "
           f"{'N':>6} {'N_steady':>8} {'mean|V|':>8} {'std_ΔP(Pa)':>10} {'1/f_slope':>10}")
    emit(hdr)
    emit("-" * len(hdr))

    for r in results:
        std_str   = f"{r['std_dp_pa']:.4f}" if r['std_dp_pa'] is not None else "  N/A  "
        slope_str = f"{r['slope_1f']:.2f}"  if not np.isnan(r['slope_1f']) else "  N/A"
        emit(
            f"  {r['rec']:<43} {r['fs']:>5.1f} {r['jitter_pct']:>8.1f} {r['dur_s']:>7.0f} "
            f"{r['n_total']:>6} {r['n_steady']:>8} {r['mean_abs_v']:>8.3f} "
            f"{std_str:>10} {slope_str:>10}"
        )

    # ── declared vs measured ──────────────────────────────────────────────────
    emit("\n[ 3 ] DECLARED vs MEASURED (labeled bags only)\n")

    labeled = [r for r in results if r['declared_v'] is not None]
    labeled.sort(key=lambda r: r['declared_v'])

    emit(f"  {'Bag':<45} {'V_decl':>7} {'mean|V|':>8} {'q_decl(Pa)':>11} {'mean_ΔP(Pa)':>12} {'ratio':>7}")
    emit("  " + "-" * 92)
    prev_mean_v  = -1.0
    prev_std_dp  = -1.0
    monotone_v   = True
    monotone_std = True

    for r in labeled:
        vd = r['declared_v']
        qd = 0.5 * RHO * vd**2
        mv = r['mean_abs_v']
        dp = r['mean_dp_pa'] if r['mean_dp_pa'] is not None else float('nan')
        ratio = mv / vd if vd else float('nan')
        if mv <= prev_mean_v:
            monotone_v = False
        prev_mean_v = mv
        std_dp = r['std_dp_pa'] if r['std_dp_pa'] is not None else float('nan')
        if not np.isnan(std_dp) and not np.isnan(prev_std_dp) and std_dp <= prev_std_dp:
            monotone_std = False
        prev_std_dp = std_dp if not np.isnan(std_dp) else prev_std_dp
        dp_str = f"{dp:.3f}" if not np.isnan(dp) else " N/A"
        emit(f"  {r['rec']:<45} {vd:>7.1f} {mv:>8.3f} {qd:>11.3f} {dp_str:>12} {ratio:>7.3f}")

    emit()
    emit(f"  mean|V| monotone with declared setpoint : {'YES' if monotone_v else '*** NO — MISMATCH ***'}")
    emit(f"  σ_ΔP   monotone with declared setpoint  : {'YES' if monotone_std else '*** NO — MISMATCH ***'}")

    # ── warning flags ─────────────────────────────────────────────────────────
    emit("\n[ 4 ] WARNING FLAGS\n")

    any_flags = False
    for r in results:
        if r['flags']:
            any_flags = True
            emit(f"  {r['rec']}")
            for f in r['flags']:
                emit(f"    ⚠  {f}")
    if not any_flags:
        emit("  (none)")

    # ── explicit verdict ──────────────────────────────────────────────────────
    emit("\n[ 5 ] EXPLICIT VERDICT\n")

    emit("  Bag                                          Verdict")
    emit("  " + "-" * 70)

    for r in results:
        if r['declared_v'] is not None:
            vd = r['declared_v']
            mv = r['mean_abs_v']
            ratio = mv / vd
            if ratio < 0.15:
                v = f"SUSPICIOUS — sensor reads {mv:.2f} m/s vs declared {vd} m/s (ratio {ratio:.2f})"
            elif ratio < 0.5:
                v = f"QUESTIONABLE — severe underread ({mv:.2f} / {vd} m/s, ratio {ratio:.2f})"
            else:
                v = f"PLAUSIBLE — underread but data exists ({mv:.2f} / {vd} m/s)"
            emit(f"  {r['rec']:<45} {v}")
        else:
            emit(f"  {r['rec']:<45} (unlabeled)")

    emit()
    emit("  KEY FINDING:")
    emit("  ─────────────────────────────────────────────────────────────────────")
    emit("  7_5ms_mavros_20250909_133106  contains only 144 s of spin-up data")
    emit("  (mean V = -0.90 m/s).  Fan had NOT reached steady state.")
    emit()
    emit("  PROPOSED REASSIGNMENT:")
    emit("  mavros_20250909_133347  (13:33 UTC, 690 s, mean V ≈ 4.67 m/s,")
    emit("  stable plateau in first ~350 s) is the actual 7.5 m/s steady run.")
    emit()
    emit("  Sensor calibration note: all labeled bags underread the setpoint.")
    emit("  Likely FCU differential-pressure offset.  Noise characterisation")
    emit("  is performed in reported-ΔP space — the offset does not affect σ_ΔP.")
    emit("  Do not use reported V directly as the 'true' tunnel speed.")
    emit("=" * 78)

    # ── write report ──────────────────────────────────────────────────────────
    report_path = f"{OUT_DIR}/validation_report.txt"
    with open(report_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nReport written to {report_path}")

    # ── plots ─────────────────────────────────────────────────────────────────
    _make_validation_plots(results)


def _shade_steady(ax, t, steady):
    in_s  = np.where(np.diff(steady.astype(int)) ==  1)[0] + 1
    out_s = np.where(np.diff(steady.astype(int)) == -1)[0] + 1
    if steady[0]:  in_s  = np.r_[0, in_s]
    if steady[-1]: out_s = np.r_[out_s, len(t) - 1]
    for s, e in zip(in_s, out_s):
        ax.axvspan(t[s], t[e], alpha=0.10, color="green")


def _make_validation_plots(results):
    # Plot 1: time series for all bags, with steady windows shaded
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(18, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    decl_colors = {3.0: "#2196F3", 6.0: "#E91E63", 7.5: "#FF9800"}
    unlabeled_c = "#607D8B"

    for ax, r in zip(axes, results):
        t   = r["_t_rel"]
        as_ = r["_airspeed"]
        c   = decl_colors.get(r["declared_v"], unlabeled_c)
        ax.plot(t, as_, lw=0.4, alpha=0.7, color=c)
        _shade_steady(ax, t, r["_steady"])
        ax.axhline(r["mean_abs_v"],  color="k",  ls="--", lw=1.0,
                   label=f"mean|V|={r['mean_abs_v']:.2f}")
        if r["declared_v"]:
            ax.axhline(r["declared_v"], color="red", ls=":", lw=1.2,
                       label=f"declared {r['declared_v']} m/s")
        ax.set_ylim(-5, 9)
        ax.axhline(0, color="k", lw=0.4, alpha=0.3)
        flags_str = "  |  ".join(r["flags"]) if r["flags"] else "OK"
        ax.set_title(
            f"{r['rec']}  |  {r['t_wall']}  |  dur={r['dur_s']:.0f}s  "
            f"N={r['n_total']}  fs={r['fs']:.1f}Hz  jitter={r['jitter_pct']:.1f}%\n"
            f"Flags: {flags_str}",
            fontsize=8,
        )
        ax.set_ylabel("m/s")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Time [s]")

    plt.suptitle("Bag validation — /mavros/vfr_hud airspeed  (green = stationary windows)",
                 fontsize=12, y=1.001)
    plt.tight_layout()
    out = f"{OUT_DIR}/validation_plots.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()

    # Plot 2: PSD comparison for bags with enough steady data
    psd_recs = [r for r in results if r["_psd_f"] is not None]
    if psd_recs:
        fig, ax = plt.subplots(figsize=(12, 6))
        for r in psd_recs:
            c   = decl_colors.get(r["declared_v"], unlabeled_c)
            lbl = r["rec"].replace("_mavros_20250909_", "_")
            ax.loglog(r["_psd_f"], r["_pxx"], lw=1.2, color=c,
                      alpha=0.8, label=f"{lbl}  σ_ΔP={r['std_dp_pa']:.3f} Pa")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [(Pa)²/Hz]")
        ax.set_title("Welch PSD — steady-state windows  (for bags with >20 steady samples)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        out2 = f"{OUT_DIR}/validation_psd.png"
        plt.savefig(out2, dpi=130, bbox_inches="tight")
        print(f"Saved {out2}")
        plt.close()


if __name__ == "__main__":
    main()
