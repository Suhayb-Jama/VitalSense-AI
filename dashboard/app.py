"""VitalSense AI dashboard. Start everything with: python run_all.py"""

import math

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

API = "http://127.0.0.1:8000"
COLORS = {"HEALTHY": "#22c55e", "WARNING": "#f59e0b", "CRITICAL": "#ef4444"}
LEVEL_COLORS = {"HIGH": "#ef4444", "LOW": "#f59e0b", "NORMAL": "#22c55e"}
ALERT_COLORS = {"CRITICAL": "#ef4444", "WARNING": "#f59e0b", "ANOMALY": "#a855f7"}

st.set_page_config(page_title="VitalSense AI", page_icon="🩺", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.kpi {background:#111827; border:1px solid #1f2937; border-radius:12px; padding:14px 18px;}
.kpi .label {color:#9ca3af; font-size:0.75rem; text-transform:uppercase; letter-spacing:.06em;}
.kpi .value {font-size:2.1rem; font-weight:700;}
.badge {display:inline-block; padding:3px 14px; border-radius:999px; font-weight:700; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)


def api_get(path):
    r = requests.get(f"{API}{path}", timeout=5)
    r.raise_for_status()
    return r.json()


def api_post(path, body):
    requests.post(f"{API}{path}", json=body, timeout=5)


# ---------------- sidebar (controls) ----------------
try:
    ids = [u["id"] for u in api_get("/equipment")["units"]]
except Exception:
    st.error("Cannot reach the VitalSense API. Start the app with: python run_all.py")
    st.stop()

st.sidebar.markdown("## VitalSense AI")
st.sidebar.caption("Predictive maintenance for data-center equipment")
selected = st.sidebar.selectbox("Select equipment", ids, key="selected")
st.sidebar.markdown("**Simulated telemetry**")
if st.sidebar.button("▶ Start degradation"):
    api_post(f"/equipment/{selected}/simulation", {"action": "start"})
if st.sidebar.button("⏸ Pause"):
    api_post(f"/equipment/{selected}/simulation", {"action": "stop"})
if st.sidebar.button("↺ Reset this unit"):
    api_post(f"/equipment/{selected}/simulation", {"action": "reset"})
if st.sidebar.button("↺ Reset all units"):
    for i in ids:
        api_post(f"/equipment/{i}/simulation", {"action": "reset"})

# ---------------- header ----------------
st.markdown("# VitalSense AI")
st.caption("AI-powered predictive maintenance · PROTOTYPE")
st.info("As me, Suhayb Jama Mohamed. I trained this model using public AI4I 2020 predictive-maintenance "
        "dataset (industrial machines), **not** real data-center data but Synthetic. This validates the AI pipeline; "
        "production would ingest real server, UPS, PDU, cooling and BMS/DCIM telemetry.")


# ---------------- sections ----------------
def render_kpis(fleet):
    items = [("Total equipment", fleet["total"], "#f9fafb"),
             ("Healthy", fleet["healthy"], COLORS["HEALTHY"]),
             ("Warning", fleet["warning"], COLORS["WARNING"]),
             ("Critical", fleet["critical"], COLORS["CRITICAL"])]
    for col, (label, value, color) in zip(st.columns(4), items):
        col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                     f'<div class="value" style="color:{color}">{value}</div></div>',
                     unsafe_allow_html=True)


def render_fleet(fleet):
    df = pd.DataFrame([{
        "Equipment": u["id"], "Class": u["type"], "Status": u["status"],
        "Risk %": u["risk_pct"], "Anomaly": "YES" if u["anomaly"] else "-",
        "Simulation": "Degrading" if u["degrading"] else "Idle",
    } for u in fleet["units"]])
    styled = df.style.map(lambda s: f"color:{COLORS.get(s, 'inherit')}; font-weight:700", subset=["Status"])
    st.dataframe(styled, hide_index=True)


def gauge(risk, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk, number={"suffix": "%"},
        title={"text": "Failure risk"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 30], "color": "#14532d"},
                         {"range": [30, 50], "color": "#78350f"},
                         {"range": [50, 100], "color": "#7f1d1d"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e5e7eb"})
    return fig


def history_figure(h):
    h = h.copy()
    h["time"] = pd.to_datetime(h["time"])
    h["process_temp_c"] = h["process_temp"] - 273.15
    panels = [("risk_pct", "Failure risk (%)", 1, 1, "#ef4444"),
              ("process_temp_c", "Equipment temp (°C)", 1, 2, "#f59e0b"),
              ("rpm", "Fan speed (rpm)", 1, 3, "#3b82f6"),
              ("torque", "Load / torque (Nm)", 2, 1, "#a855f7"),
              ("power_w", "Power (W)", 2, 2, "#14b8a6"),
              ("tool_wear", "Equipment wear (min)", 2, 3, "#e5e7eb")]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[p[1] for p in panels])
    for col, _, r, c, color in panels:
        fig.add_trace(go.Scatter(x=h["time"], y=h[col], mode="lines",
                                 line=dict(color=color, width=2), showlegend=False), row=r, col=c)
    fig.add_hline(y=30, line_dash="dot", line_color=COLORS["WARNING"], row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color=COLORS["CRITICAL"], row=1, col=1)
    fig.update_layout(height=430, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def render_unit(d, hist):
    color = COLORS[d["status"]]
    st.markdown(f"### {d['equipment_id']} &nbsp; "
                f"<span class='badge' style='background:{color}22;color:{color};border:1px solid {color}'>"
                f"{d['status']}</span>", unsafe_allow_html=True)

    left, right = st.columns([1, 2])
    with left:
        st.plotly_chart(gauge(d["risk_pct"], color))
        if d["anomaly"]["is_anomaly"]:
            st.error("Anomaly detected: readings are unusual versus normal behavior")
        else:
            st.success("No anomaly detected")
    with right:
        r = d["reading"]
        power = r["torque"] * r["rpm"] * 2 * math.pi / 60
        tiles = [("Equipment temp", f"{r['process_temp'] - 273.15:.1f} °C"),
                 ("Ambient temp", f"{r['air_temp'] - 273.15:.1f} °C"),
                 ("Fan speed", f"{r['rpm']:.0f} rpm"),
                 ("Load (torque)", f"{r['torque']:.1f} Nm"),
                 ("Power (derived)", f"{power:,.0f} W"),
                 ("Equipment wear", f"{r['tool_wear']:.0f} min")]
        for row in (tiles[:3], tiles[3:]):
            for col, (label, value) in zip(st.columns(3), row):
                col.metric(label, value)

    if len(hist) > 1:
        st.plotly_chart(history_figure(hist))

    st.markdown("#### AI explanation")
    c1, c2 = st.columns(2)
    with c1:
        sig = pd.DataFrame(d["signals"]).rename(columns={"name": "Signal", "display": "Value", "level": "Level"})
        st.dataframe(sig.style.map(lambda s: f"color:{LEVEL_COLORS.get(s, 'inherit')}; font-weight:700",
                                   subset=["Level"]), hide_index=True)
    with c2:
        contrib = pd.DataFrame(d["contributions"])
        contrib = contrib[contrib["impact"] > 0]
        if contrib.empty:
            st.caption("No signal is pushing the risk up.")
        else:
            fig = go.Figure(go.Bar(x=contrib["impact"], y=contrib["signal"], orientation="h",
                                   marker_color=color))
            fig.update_layout(height=220, template="plotly_dark", title="Model contribution (risk points)",
                              margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig)

    st.markdown(f"**Possible cause:** {d['cause']}")
    box = {"HEALTHY": st.success, "WARNING": st.warning, "CRITICAL": st.error}[d["status"]]
    box(f"**Recommended action:** {d['recommendation']}")
    st.caption(d["disclaimer"])


def render_alerts(alerts):
    st.markdown("#### Alerts")
    if not alerts:
        st.caption("No alerts yet.")
        return
    df = pd.DataFrame(alerts)[["time", "equipment", "level", "message"]]
    df.columns = ["Time", "Equipment", "Level", "Message"]
    st.dataframe(df.head(10).style.map(
        lambda s: f"color:{ALERT_COLORS.get(s, 'inherit')}; font-weight:700", subset=["Level"]),
        hide_index=True)


@st.fragment(run_every=2)
def live_view():
    try:
        fleet = api_get("/equipment")
        sel = st.session_state["selected"]
        detail = api_get(f"/equipment/{sel}")
        hist = pd.DataFrame(api_get(f"/equipment/{sel}/history"))
        alerts = api_get("/alerts")
    except Exception as e:
        st.warning(f"Waiting for API... ({e})")
        return
    render_kpis(fleet)
    st.markdown("#### Fleet overview")
    render_fleet(fleet)
    render_unit(detail, hist)
    render_alerts(alerts)


live_view()

with st.expander("How this maps to real data-center telemetry"):
    st.table(pd.DataFrame({
        "Prototype signal (AI4I 2020)": ["Air temperature", "Process temperature", "Rotational speed",
                                         "Torque", "Power (derived)", "Tool wear", "Machine failure"],
        "Future data-center source": ["Ambient / cold-aisle temperature (BMS, sensors)",
                                      "Server / component temperature (server BMC, DCIM)",
                                      "Fan / cooling-unit speed (servers, CRAC/CRAH)",
                                      "CPU / workload load signal",
                                      "Rack / PDU / UPS power draw",
                                      "Runtime hours / component age",
                                      "Equipment failure or outage tickets"]}))