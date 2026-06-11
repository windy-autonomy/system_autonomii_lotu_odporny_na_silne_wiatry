"""
ui_app.py — interfejs graficzny stanowiska matrycy Pitota (prototyp/demo).

Uruchomienie:
  python3 ui_app.py
  Otwórz: http://127.0.0.1:8050

Zakładki:
  - Monitor    — replay baga ROS2: airspeed (top) + IMU pitch (bottom)
  - Trajektoria — projektant profilu schodkowego i sweepowego
  - Rejestracja — zarządzanie nagraniami (ROS2 bag)
  - Analiza     — krzywe RMSE z results/experiments.json
"""

import json
import os
import sqlite3
import struct

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dash_table, dcc, html
from plotly.subplots import make_subplots

# ── paleta kolorów ─────────────────────────────────────────────────────────────
GREEN  = "#1D9E75"
RED    = "#E24B4A"
BLUE   = "#378ADD"
AMBER  = "#BA7517"
VIOLET = "#7B4FB7"
GRAY   = "#6C757D"
DARK   = "#2C3E50"

# ── ładowanie danych z baga ROS2 ───────────────────────────────────────────────
_BAG_PATH = "windwall/6ms_mavros_20250909_131341/mavros_20250909_131341_0.db3"
_WINDOW_S = 30.0     # szerokość okna replay [s]
_DT_S     = 1.0      # krok replay na jeden tick [s]


def _decode_vfrhud(blob):
    b = bytes(blob)
    off = 4
    sec, ns = struct.unpack_from("<II", b, off); off += 8
    fl = struct.unpack_from("<I", b, off)[0]; off += 4 + fl
    if off % 4: off += 4 - off % 4
    return sec + ns * 1e-9, struct.unpack_from("<f", b, off)[0]


def _decode_imu(blob):
    b = bytes(blob)
    off = 4
    sec, ns = struct.unpack_from("<II", b, off); off += 8
    fl = struct.unpack_from("<I", b, off)[0]; off += 4 + fl
    if off % 4: off += 4 - off % 4
    qx, qy, qz, qw = struct.unpack_from("<4d", b, off); off += 32 + 72
    avx, avy, avz   = struct.unpack_from("<3d", b, off)
    t     = sec + ns * 1e-9
    pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0)))
    return t, pitch, np.degrees(avy)


def _load_bag():
    """Ładuje wszystkie wiadomości airspeed + IMU z baga. Zwraca dict lub None."""
    if not os.path.exists(_BAG_PATH):
        return None
    try:
        conn = sqlite3.connect(_BAG_PATH)

        tid_as = conn.execute(
            "SELECT id FROM topics WHERE name='/mavros/vfr_hud'"
        ).fetchone()[0]
        rows_as = conn.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (tid_as,),
        ).fetchall()
        decoded_as = [_decode_vfrhud(r[0]) for r in rows_as]
        t_as = np.array([d[0] for d in decoded_as])
        v_as = np.array([d[1] for d in decoded_as])

        tid_im = conn.execute(
            "SELECT id FROM topics WHERE name='/mavros/imu/data'"
        ).fetchone()[0]
        rows_im = conn.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (tid_im,),
        ).fetchall()
        decoded_im = [_decode_imu(r[0]) for r in rows_im]
        t_im    = np.array([d[0] for d in decoded_im])
        pitch   = np.array([d[1] for d in decoded_im])
        pitchr  = np.array([d[2] for d in decoded_im])

        conn.close()

        t0   = min(t_as[0], t_im[0])
        t_as -= t0
        t_im -= t0
        dur   = max(t_as[-1], t_im[-1])

        print(f"[bag] loaded — airspeed: {len(t_as)} msg ({dur:.0f} s), "
              f"IMU: {len(t_im)} msg")
        return dict(t_as=t_as, v_as=v_as, t_im=t_im,
                    pitch=pitch, pitchr=pitchr, dur=dur)
    except Exception as exc:
        print(f"[bag] load error: {exc}")
        return None


_BAG = _load_bag()
_DUR = _BAG["dur"] if _BAG else 60.0


# ── figure: monitor z danymi z baga ───────────────────────────────────────────

def _fig_monitor(n: int) -> go.Figure:
    """Horizontal split: airspeed (góra) + IMU pitch (dół). Okno 30 s."""
    t_now = _WINDOW_S + (n % int((_DUR - _WINDOW_S) or 1))
    t0    = t_now - _WINDOW_S

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=[
            "Prędkość napływu  ·  /mavros/vfr_hud  [m/s]",
            "Pochylenie (IMU pitch)  ·  /mavros/imu/data  [°]",
        ],
    )

    if _BAG:
        d  = _BAG
        ma = (d["t_as"] >= t0) & (d["t_as"] <= t_now)
        mi = (d["t_im"] >= t0) & (d["t_im"] <= t_now)

        fig.add_trace(go.Scatter(
            x=d["t_as"][ma], y=d["v_as"][ma],
            mode="lines", name="V [m/s]",
            line=dict(color=AMBER, width=2.0),
        ), row=1, col=1)

        if ma.sum() > 1:
            mu = float(np.nanmean(d["v_as"][ma]))
            fig.add_hline(y=mu, line_dash="dot", line_color=GREEN, line_width=1.4,
                          annotation_text=f"śr. {mu:.2f} m/s",
                          annotation_position="bottom right", row=1, col=1)

        fig.add_trace(go.Scatter(
            x=d["t_im"][mi], y=d["pitch"][mi],
            mode="lines", name="pitch [°]",
            line=dict(color=BLUE, width=2.0),
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=d["t_im"][mi], y=d["pitchr"][mi],
            mode="lines", name="pitch rate [°/s]",
            line=dict(color=VIOLET, width=1.0), opacity=0.55,
        ), row=2, col=1)

    else:
        rng = np.random.default_rng(n % 1000)
        t_s = np.linspace(t0, t_now, 300)
        fig.add_trace(go.Scatter(x=t_s, y=4.0 + rng.normal(0, 0.2, 300),
                                 mode="lines", name="V [m/s]",
                                 line=dict(color=AMBER, width=2.0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_s, y=rng.normal(-1.6, 0.05, 300),
                                 mode="lines", name="pitch [°]",
                                 line=dict(color=BLUE, width=2.0)), row=2, col=1)

    fig.update_xaxes(title_text="czas  [s]", row=2, col=1,
                     showgrid=True, gridcolor="#EBEBEB",
                     range=[t0, t_now])
    fig.update_xaxes(showgrid=True, gridcolor="#EBEBEB", row=1, col=1)
    fig.update_yaxes(title_text="V  [m/s]",    row=1, col=1,
                     showgrid=True, gridcolor="#EBEBEB")
    fig.update_yaxes(title_text="pitch  [°]",  row=2, col=1,
                     showgrid=True, gridcolor="#EBEBEB")
    fig.update_layout(
        height=540,
        margin=dict(l=55, r=20, t=42, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=11),
    )
    return fig


# ── inne figure generators (trajektoria, analiza) ──────────────────────────────

def _default_steps():
    return [
        {"krok": i + 1, "alpha_deg": a, "beta_deg": 0, "dwell_s": 5}
        for i, a in enumerate([-20, -10, 0, 10, 20, 10, 0, -10, -20])
    ]


def _fig_step_preview(rows: list) -> go.Figure:
    fig = go.Figure()
    if not rows:
        return fig
    t_pts, a_pts, b_pts = [0.0], [], []
    for row in rows:
        try:
            a = float(row.get("alpha_deg", 0) or 0)
            b = float(row.get("beta_deg",  0) or 0)
            d = float(row.get("dwell_s",   5) or 5)
        except (TypeError, ValueError):
            continue
        a_pts += [a, a]; b_pts += [b, b]
        t_pts += [t_pts[-1], t_pts[-1] + d]
    t_pts = t_pts[:-1]
    if not a_pts:
        return fig
    fig.add_trace(go.Scatter(
        x=t_pts, y=a_pts, mode="lines",
        line=dict(color=BLUE, width=2.5),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.10)",
        name="α zadane",
    ))
    fig.add_trace(go.Scatter(
        x=t_pts, y=b_pts, mode="lines",
        line=dict(color=VIOLET, width=1.8, dash="dot"),
        name="β zadane",
    ))
    total = sum(float(r.get("dwell_s", 5) or 5) for r in rows)
    fig.add_annotation(
        text=f"Czas całkowity: <b>{total:.0f} s</b>",
        xref="paper", yref="paper", x=0.98, y=0.96,
        showarrow=False, align="right", font=dict(size=11, color=DARK),
    )
    fig.update_layout(
        height=300, margin=dict(l=50, r=20, t=20, b=45),
        xaxis_title="czas  [s]", yaxis_title="kąt  [°]",
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.04, x=0), font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EBEBEB")
    fig.update_yaxes(showgrid=True, gridcolor="#EBEBEB",
                     zeroline=True, zerolinecolor="#CCCCCC")
    return fig


def _fig_sweep_preview(a_min=-20, a_max=20, rate=5, n_cyc=2) -> go.Figure:
    span   = float(a_max) - float(a_min)
    rate   = max(float(rate), 0.1)
    t_half = span / rate
    t_tot  = 2 * t_half * int(n_cyc)
    t_s    = np.linspace(0, t_tot, 800)
    phase  = (t_s % (2 * t_half)) / (2 * t_half)
    alpha  = np.where(phase < 0.5,
                      float(a_min) + 2 * phase * span,
                      float(a_max) - 2 * (phase - 0.5) * span)
    fig = go.Figure(go.Scatter(
        x=t_s, y=alpha, mode="lines",
        line=dict(color=VIOLET, width=2.5),
        fill="tozeroy", fillcolor="rgba(123,79,183,0.09)",
        name=f"α  [{a_min}° → {a_max}°]",
    ))
    fig.add_annotation(
        text=(f"Czas całkowity: <b>{t_tot:.0f} s</b>   ·   "
              f"Prędkość: <b>{rate:.0f}°/s</b>   ·   Cykle: <b>{n_cyc}</b>"),
        xref="paper", yref="paper", x=0.5, y=0.96,
        showarrow=False, font=dict(size=11, color=DARK),
    )
    fig.update_layout(
        height=300, margin=dict(l=50, r=20, t=20, b=45),
        xaxis_title="czas  [s]", yaxis_title="α  [°]",
        plot_bgcolor="white", paper_bgcolor="white", font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EBEBEB")
    fig.update_yaxes(showgrid=True, gridcolor="#EBEBEB",
                     zeroline=True, zerolinecolor="#CCCCCC")
    return fig


def _fig_analysis() -> go.Figure:
    try:
        with open("results/experiments.json") as f:
            r = json.load(f)
        sp     = r["rmse_vs_speed"]["speed"]
        rmse_a = r["rmse_vs_speed"]["rmse_a"]
        rmse_b = r["rmse_vs_speed"]["rmse_b"]
        rmse_V = r["rmse_vs_speed"]["rmse_V"]
        red    = r["filtering"]["reduction"]
        rmse_af = [a / x for a, x in zip(rmse_a, red)]
    except Exception:
        sp = [3.0, 6.0, 7.5]; rmse_a = [10.60, 2.18, 1.41]
        rmse_b = [10.53, 2.10, 1.31]; rmse_V = [0.451, 0.151, 0.120]
        rmse_af = [2.73, 0.56, 0.36]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["RMSE(α)  [°]", "RMSE(β)  [°]",
                                        "RMSE(V)  [m/s]"])
    for col, (vals, fv, clr) in enumerate(
        [(rmse_a, rmse_af, RED), (rmse_b, None, BLUE), (rmse_V, None, GREEN)],
        start=1,
    ):
        fig.add_trace(go.Scatter(
            x=sp, y=vals, mode="lines+markers+text",
            line=dict(color=clr, width=2.2), marker=dict(size=10),
            text=[f"{v:.2f}" for v in vals], textposition="top center",
            name="surowe" if col == 1 else "", showlegend=(col == 1),
        ), row=1, col=col)
        if fv:
            fig.add_trace(go.Scatter(
                x=sp, y=fv, mode="lines+markers+text",
                line=dict(color=GREEN, width=2.0, dash="dash"), marker=dict(size=9),
                text=[f"{v:.2f}" for v in fv], textposition="bottom center",
                name="po filtracji", showlegend=(col == 1),
            ), row=1, col=col)
    fig.update_xaxes(tickvals=sp, title_text="V  [m/s]",
                     showgrid=True, gridcolor="#EBEBEB")
    fig.update_yaxes(showgrid=True, gridcolor="#EBEBEB", rangemode="tozero")
    fig.update_layout(
        height=380, margin=dict(l=50, r=20, t=55, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=1.12, x=0), font=dict(size=11),
    )
    return fig


# ── nagłówek ───────────────────────────────────────────────────────────────────

def _badge(label, val, color):
    return dbc.Badge(
        [html.Span(label + ": "), html.Span(val, style={"fontWeight": 700})],
        color=color, className="me-2 py-2 px-3 fs-6",
    )


_header = dbc.Row([
    dbc.Col([
        html.H5("🔬  Stanowisko Matrycy Pitota — Estymacja Wiatru",
                className="mb-0 fw-bold", style={"color": DARK}),
        html.Small("Windwall + ramię robotyczne · 5×Pitot · Czujnik F/T 6-osi · ROS 2 Humble",
                   className="text-muted"),
    ], width=6),
    dbc.Col([
        _badge("Robot",    "POŁĄCZONY", "success"),
        _badge("Struga",   "6,0 m/s",   "primary"),
        _badge("Nagranie", "STOP",       "secondary"),
        _badge("Bag", "6ms · 3728 msg", "light"),
    ], width=6,
       className="d-flex align-items-center justify-content-end flex-wrap gap-1"),
], className="py-3 border-bottom mb-3")


# ── karty estymaty (prawa kolumna) ─────────────────────────────────────────────

def _est_card(label, value, unit, color):
    return dbc.Card(dbc.CardBody([
        html.Small(label, className="text-muted d-block mb-1"),
        html.H3(value, style={"color": color, "fontWeight": 700,
                              "marginBottom": 2, "fontSize": "1.5rem"}),
        html.Small(unit, className="text-muted"),
    ], className="py-2 px-3"), className="mb-2 shadow-sm")


# ── ZAKŁADKA 1: Monitor ────────────────────────────────────────────────────────

_bag_info = (
    "6ms · 3 727 msg airspeed · 18 637 msg IMU"
    if _BAG else "bag niedostępny — dane syntetyczne"
)

_tab_monitor = dbc.Tab(
    label="📡  Monitor", tab_id="tab-monitor",
    children=dbc.Row([
        # ── main plots ───────────────────────────────────────────────────────
        dbc.Col([
            html.Small(f"Replay baga ROS2: {_bag_info}", className="text-muted d-block mb-1"),
            dcc.Graph(id="graph-monitor", figure=_fig_monitor(0),
                      config={"displayModeBar": False}),
        ], width=9),
        # ── karty statusu ────────────────────────────────────────────────────
        dbc.Col([
            html.H6("Bieżące odczyty", className="text-muted mb-2 mt-1"),
            _est_card("Prędkość V",       "3,89 m/s", "vfr_hud  (śr.)",   AMBER),
            _est_card("Pochylenie pitch",  "−1,60°",   "IMU quat → Euler", BLUE),
            _est_card("Pitch rate",        "0,03 °/s", "ang. vel. y",      VIOLET),
            html.Hr(className="my-2"),
            html.H6("Parametry baga", className="text-muted mb-2"),
            dbc.Table([html.Tbody([
                html.Tr([html.Td("Setpoint",     className="text-muted small"),
                         html.Td("6,0 m/s",      className="fw-semibold small")]),
                html.Tr([html.Td("Czas trwania", className="text-muted small"),
                         html.Td(f"{_DUR:.0f} s",className="fw-semibold small")]),
                html.Tr([html.Td("f_s airspeed", className="text-muted small"),
                         html.Td("≈10 Hz",       className="fw-semibold small")]),
                html.Tr([html.Td("f_s IMU",      className="text-muted small"),
                         html.Td("≈50 Hz",       className="fw-semibold small")]),
            ])], size="sm", bordered=False, striped=True),
            html.Hr(className="my-2"),
            html.Small("Odtwarzanie ciągłe · pętla", className="text-muted"),
        ], width=3),
    ], className="mt-2"),
)


# ── ZAKŁADKA 2: Trajektoria ────────────────────────────────────────────────────

_step_columns = [
    {"name": "#",     "id": "krok",      "editable": False, "type": "numeric"},
    {"name": "α [°]", "id": "alpha_deg", "editable": True,  "type": "numeric"},
    {"name": "β [°]", "id": "beta_deg",  "editable": True,  "type": "numeric"},
    {"name": "t [s]", "id": "dwell_s",   "editable": True,  "type": "numeric"},
]

_tab_trajectory = dbc.Tab(
    label="🔁  Trajektoria", tab_id="tab-trajectory",
    children=dbc.Tabs([
        dbc.Tab(label="Profil schodkowy", tab_id="traj-step",
                children=dbc.Row([
                    dbc.Col([
                        html.P("Sekwencja kątów — robot zatrzymuje się na każdym poziomie.",
                               className="text-muted small mb-2"),
                        dash_table.DataTable(
                            id="table-steps",
                            columns=_step_columns,
                            data=_default_steps(),
                            row_deletable=True,
                            style_header={"fontWeight": "bold",
                                          "backgroundColor": "#f8f9fa",
                                          "borderBottom": "2px solid #dee2e6"},
                            style_cell={"padding": "8px 14px", "fontSize": "13px",
                                        "fontFamily": "monospace"},
                            style_data_conditional=[
                                {"if": {"column_id": "krok"}, "color": GRAY},
                                {"if": {"column_id": "alpha_deg",
                                        "filter_query": "{alpha_deg} > 0"},
                                 "color": GREEN, "fontWeight": "bold"},
                                {"if": {"column_id": "alpha_deg",
                                        "filter_query": "{alpha_deg} < 0"},
                                 "color": RED, "fontWeight": "bold"},
                            ],
                        ),
                        dbc.ButtonGroup([
                            dbc.Button("＋ Dodaj krok", id="btn-add-step",
                                       color="outline-secondary", size="sm"),
                            dbc.Button("Eksportuj JSON",
                                       color="outline-primary", size="sm"),
                            dbc.Button("▶  Wyślij do robota",
                                       color="success", size="sm", disabled=True),
                        ], className="mt-3"),
                    ], width=5),
                    dbc.Col([
                        html.H6("Podgląd trajektorii", className="text-muted mb-1"),
                        dcc.Graph(id="graph-step-preview",
                                  figure=_fig_step_preview(_default_steps()),
                                  config={"displayModeBar": False}),
                        dbc.Alert(id="alert-step-info",
                                  children="9 kroków · czas całkowity: 45 s",
                                  color="info", className="mt-2 py-2 small"),
                    ], width=7),
                ], className="mt-3"),
        ),
        dbc.Tab(label="Sweep ciągły", tab_id="traj-sweep",
                children=dbc.Row([
                    dbc.Col([
                        html.P("Ciągłe przemiatanie ze stałą prędkością kątową.",
                               className="text-muted small mb-3"),
                        html.Label("α_min  [°]", className="fw-semibold"),
                        dcc.Slider(id="sl-amin", min=-30, max=0, step=1, value=-20,
                                   marks={i: f"{i}°" for i in range(-30, 1, 10)},
                                   className="mb-3"),
                        html.Label("α_max  [°]", className="fw-semibold"),
                        dcc.Slider(id="sl-amax", min=0, max=30, step=1, value=20,
                                   marks={i: f"{i}°" for i in range(0, 31, 10)},
                                   className="mb-3"),
                        html.Label("Prędkość kątowa  [°/s]", className="fw-semibold"),
                        dcc.Slider(id="sl-rate", min=1, max=20, step=1, value=5,
                                   marks={i: f"{i}" for i in [1, 5, 10, 15, 20]},
                                   className="mb-3"),
                        html.Label("Liczba cykli", className="fw-semibold"),
                        dcc.Slider(id="sl-ncyc", min=1, max=6, step=1, value=2,
                                   marks={i: str(i) for i in range(1, 7)},
                                   className="mb-4"),
                        dbc.ButtonGroup([
                            dbc.Button("Eksportuj JSON",
                                       color="outline-primary", size="sm"),
                            dbc.Button("▶  Wyślij do robota",
                                       color="success", size="sm", disabled=True),
                        ]),
                    ], width=4),
                    dbc.Col([
                        html.H6("Podgląd sweepа", className="text-muted mb-1"),
                        dcc.Graph(id="graph-sweep-preview",
                                  figure=_fig_sweep_preview(),
                                  config={"displayModeBar": False}),
                    ], width=8),
                ], className="mt-3"),
        ),
    ], id="traj-subtabs", active_tab="traj-step"),
)


# ── ZAKŁADKA 3: Rejestracja ────────────────────────────────────────────────────

_mock_runs = [
    {"id": "RUN-001", "V": "3,0 m/s", "trajektoria": "Schodkowy ±20°",
     "start": "09:13:02", "czas": "120 s", "próbki": "1 200",
     "bag": "run_001_0.db3", "status": "✅ OK"},
    {"id": "RUN-002", "V": "6,0 m/s", "trajektoria": "Schodkowy ±20°",
     "start": "09:16:45", "czas": "120 s", "próbki": "1 200",
     "bag": "run_002_0.db3", "status": "✅ OK"},
    {"id": "RUN-003", "V": "7,5 m/s", "trajektoria": "Sweep ±25°, 2 cykle",
     "start": "09:21:10", "czas": "80 s",  "próbki": "800",
     "bag": "run_003_0.db3", "status": "✅ OK"},
    {"id": "RUN-004", "V": "6,0 m/s", "trajektoria": "Schodkowy ±20°",
     "start": "09:28:33", "czas": "—",    "próbki": "—",
     "bag": "—", "status": "⏺ TRWA"},
]

_tab_recording = dbc.Tab(
    label="⏺  Rejestracja", tab_id="tab-recording",
    children=dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("Parametry nagrania", className="mb-0 fw-bold")),
            dbc.CardBody([
                html.Label("Nazwa / etykieta", className="fw-semibold"),
                dbc.Input(value="RUN-004", type="text", className="mb-3 font-monospace"),
                html.Label("Prędkość zadana V  [m/s]", className="fw-semibold"),
                dbc.Select(options=[{"label": f"{v} m/s", "value": str(v)}
                                    for v in [3.0, 6.0, 7.5]],
                           value="6.0", className="mb-3"),
                html.Label("Trajektoria", className="fw-semibold"),
                dbc.Select(options=[
                    {"label": "Schodkowy ±20° · 9 poziomów × 5 s", "value": "step"},
                    {"label": "Sweep ciągły ±20° · 2 cykle · 5°/s",  "value": "sweep"},
                    {"label": "(brak — ręczna)",                       "value": "manual"},
                ], value="step", className="mb-4"),
                html.Label("Katalog zapisu  (ROS2 bag)", className="fw-semibold"),
                dbc.Input(value="windwall/RUN-004/", type="text",
                          className="mb-4 font-monospace",
                          style={"fontSize": "13px"}),
                dbc.Row([
                    dbc.Col(dbc.Button("⏺  Rozpocznij", color="danger",
                                       className="w-100"), width=6),
                    dbc.Col(dbc.Button("⏹  Zatrzymaj", color="secondary",
                                       className="w-100", disabled=True), width=6),
                ]),
            ]),
        ], className="shadow-sm"), width=4),
        dbc.Col([
            html.H6("Historia nagrań", className="text-muted mb-2"),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in
                         ["id", "V", "trajektoria", "start", "czas",
                          "próbki", "bag", "status"]],
                data=_mock_runs,
                style_header={"fontWeight": "bold", "backgroundColor": "#f8f9fa",
                              "borderBottom": "2px solid #dee2e6"},
                style_cell={"padding": "8px 12px", "fontSize": "13px"},
                style_data_conditional=[
                    {"if": {"filter_query": '{status} contains "TRWA"'},
                     "backgroundColor": "#FFF3CD", "fontWeight": "bold"},
                    {"if": {"column_id": "status",
                            "filter_query": '{status} contains "OK"'},
                     "color": GREEN},
                ],
                row_selectable="single",
            ),
        ], width=8),
    ], className="mt-3"),
)


# ── ZAKŁADKA 4: Analiza ────────────────────────────────────────────────────────

_tab_analysis = dbc.Tab(
    label="📊  Analiza", tab_id="tab-analysis",
    children=[
        dbc.Row([dbc.Col([
            html.H6("Błąd estymacji RMSE — Monte-Carlo 400 realizacji/punkt",
                    className="text-muted mb-1"),
            dcc.Graph(figure=_fig_analysis(), config={"displayModeBar": False}),
        ])], className="mt-2"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Kluczowe liczby (MC @ σ = 1,05 Pa)", className="fw-bold mb-2"),
                dbc.Table([
                    html.Thead(html.Tr([html.Th(h) for h in
                                        ["V", "RMSE(α)", "RMSE(β)", "RMSE(V)", "po KF"]])),
                    html.Tbody([
                        html.Tr([html.Td("3,0 m/s"),
                                 html.Td("10,60°", style={"color": RED}),
                                 html.Td("10,53°", style={"color": RED}),
                                 html.Td("0,451 m/s"),
                                 html.Td("2,73°", style={"color": GREEN})]),
                        html.Tr([html.Td("6,0 m/s"), html.Td("2,18°"),
                                 html.Td("2,10°"), html.Td("0,151 m/s"),
                                 html.Td("0,56°", style={"color": GREEN})]),
                        html.Tr([html.Td("7,5 m/s"),
                                 html.Td("1,41°", style={"color": GREEN}),
                                 html.Td("1,31°", style={"color": GREEN}),
                                 html.Td("0,120 m/s"),
                                 html.Td("0,36°", style={"color": GREEN})]),
                    ]),
                ], bordered=True, striped=True, size="sm", className="mb-0"),
            ]), className="shadow-sm"), width=8),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Pliki wynikowe", className="fw-bold mb-2"),
                *[html.P(f, className="font-monospace small mb-1") for f in [
                    "results/experiments.json",
                    "SKNTI/img/pitot_rmse_vs_speed.png",
                    "SKNTI/img/pitot_rmse_all.png",
                    "SKNTI/img/degradation_panel.png",
                    "SKNTI/img/error_maps_ab.png",
                ]],
                dbc.Button("Przelicz wyniki", color="outline-secondary",
                           size="sm", className="mt-2", disabled=True),
            ]), className="shadow-sm"), width=4),
        ], className="mt-3"),
    ],
)


# ── aplikacja ──────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Stanowisko Matrycy Pitota"

app.layout = dbc.Container([
    # Interval i Store poza zakładkami — zawsze w DOM
    dcc.Interval(id="interval-monitor", interval=1000, n_intervals=0),
    _header,
    dbc.Tabs([_tab_monitor, _tab_trajectory, _tab_recording, _tab_analysis],
             id="main-tabs", active_tab="tab-monitor"),
    html.Hr(className="mt-4"),
    html.Small("Stanowisko badawcze  ·  Windwall + ramię robotyczne  ·  2025–2026",
               className="text-muted d-block text-center mb-3"),
], fluid=True, className="px-4")


# ── callbacks ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("graph-monitor", "figure"),
    Input("interval-monitor", "n_intervals"),
)
def _update_monitor(n):
    return _fig_monitor(n or 0)


@app.callback(
    Output("table-steps", "data"),
    Input("btn-add-step", "n_clicks"),
    State("table-steps", "data"),
    prevent_initial_call=True,
)
def _add_step(_clicks, current):
    current = list(current or [])
    current.append({"krok": len(current) + 1,
                    "alpha_deg": 0, "beta_deg": 0, "dwell_s": 5})
    for i, row in enumerate(current):
        row["krok"] = i + 1
    return current


@app.callback(
    Output("graph-step-preview", "figure"),
    Output("alert-step-info",    "children"),
    Input("table-steps", "data"),
)
def _update_step_preview(data):
    data  = data or []
    total = sum(float(r.get("dwell_s") or 5) for r in data)
    info  = f"{len(data)} kroków  ·  czas całkowity: {total:.0f} s"
    return _fig_step_preview(data), info


@app.callback(
    Output("graph-sweep-preview", "figure"),
    Input("sl-amin", "value"),
    Input("sl-amax", "value"),
    Input("sl-rate",  "value"),
    Input("sl-ncyc",  "value"),
)
def _update_sweep(a_min, a_max, rate, n_cyc):
    return _fig_sweep_preview(a_min, a_max, rate, n_cyc)


# ── uruchomienie ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
