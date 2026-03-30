"""
Module 7: Streamlit Dashboard (Frontend)
Run with: streamlit run dashboard.py
"""

import time
import threading
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# ── Project imports ──────────────────────────────────────
from database import (
    initialize_database,
    fetch_recent_traffic,
    fetch_recent_anomalies,
    fetch_incident_timeline,
    get_summary_stats,
)
from ai_model   import load_model, predict_packet
from capture    import start_simulation, stop_simulation, is_simulation_running
from incident_reconstruction import reconstruct_incidents, generate_incident_report

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="NetGrad",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}
code, pre, .stCode {
    font-family: 'Space Mono', monospace !important;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1424 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a2e 0%, #091220 100%);
    border-right: 1px solid #1e3a5f;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0f1e35 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 24px rgba(0,100,255,0.08);
}
div[data-testid="metric-container"] > label {
    color: #64b5f6 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] > div {
    color: #e2e8f0 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    font-family: 'Space Mono', monospace !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e3a5f;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64b5f6;
    border: 1px solid transparent;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: #0f1e35 !important;
    border-color: #1e3a5f !important;
    color: #90caf9 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
    color: #e3f2fd;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.04em;
    padding: 10px 24px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%);
    box-shadow: 0 0 16px rgba(21,101,192,0.5);
    transform: translateY(-1px);
}

/* ── Dataframes ── */
.stDataFrame {
    border: 1px solid #1e3a5f !important;
    border-radius: 10px;
}

/* ── Status badge helpers ── */
.badge-high   { background:#c62828; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; }
.badge-medium { background:#f57f17; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; }
.badge-low    { background:#2e7d32; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; }
.badge-none   { background:#1565c0; color:#fff; padding:2px 8px; border-radius:12px; font-size:0.75rem; font-weight:700; }

/* ── Section headers ── */
h1 { color: #90caf9 !important; font-family: 'Space Mono', monospace !important; letter-spacing:-0.02em; }
h2 { color: #64b5f6 !important; font-weight: 600 !important; }
h3 { color: #4fc3f7 !important; }

/* ── Alert box ── */
.alert-anomaly {
    background: linear-gradient(90deg,#3b1a1a,#1a0a0a);
    border-left: 4px solid #ef5350;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 4px 0;
    font-family: 'Space Mono', monospace;
    font-size:0.8rem;
    color:#ffcdd2;
}
.alert-normal {
    background: linear-gradient(90deg,#0a1f0a,#061206);
    border-left: 4px solid #66bb6a;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 4px 0;
    font-family: 'Space Mono', monospace;
    font-size:0.8rem;
    color:#c8e6c9;
}
</style>
""", unsafe_allow_html=True)


# ── Initialise ───────────────────────────────────────────
initialize_database()

@st.cache_resource
def get_model():
    return load_model()

model, scaler = get_model()


# ── Session-state defaults ────────────────────────────────
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False
if "live_packets" not in st.session_state:
    st.session_state.live_packets = []
if "packet_callback_registered" not in st.session_state:
    st.session_state.packet_callback_registered = False


# ── Packet callback ───────────────────────────────────────
from database import insert_traffic_log, insert_anomaly_record

def _handle_packet(pkt):
    result = predict_packet(pkt, model, scaler)
    pkt["is_anomaly"]    = result["is_anomaly"]
    pkt["anomaly_score"] = result["anomaly_score"]
    insert_traffic_log(pkt)
    if result["is_anomaly"]:
        insert_anomaly_record({
            "timestamp":     pkt.get("timestamp", datetime.now().isoformat()),
            "src_ip":        pkt.get("src_ip", ""),
            "dst_ip":        pkt.get("dst_ip", ""),
            "protocol":      pkt.get("protocol", ""),
            "anomaly_score": result["anomaly_score"],
            "severity":      result["severity"],
            "description":   result["description"],
            "raw_features":  result["features"],
        })


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Control Panel")
    st.markdown("---")

    st.markdown("### ⚙️ Simulation Settings")
    pkt_rate     = st.slider("Packets / second", 0.5, 10.0, 2.0, 0.5)
    attack_ratio = st.slider("Attack ratio",     0.01, 0.50, 0.08, 0.01)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start", use_container_width=True):
            if not st.session_state.pipeline_running:
                start_simulation(_handle_packet, pkt_rate, attack_ratio)
                st.session_state.pipeline_running = True
                st.success("Pipeline started!")
    with col2:
        if st.button("■ Stop", use_container_width=True):
            stop_simulation()
            st.session_state.pipeline_running = False
            st.warning("Pipeline stopped.")

    st.markdown("---")
    status_color = "🟢" if st.session_state.pipeline_running else "🔴"
    st.markdown(f"**Status:** {status_color} {'Running' if st.session_state.pipeline_running else 'Stopped'}")

    auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
    if auto_refresh and st.session_state.pipeline_running:
        time.sleep(5)
        st.rerun()

    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption("AI Network Anomaly Detection v1.0")


# ── Main title ────────────────────────────────────────────
st.markdown("# 🛡️ NetGrad")
st.caption("AI Network Anomaly Detection System")
st.markdown("---")


# ── Summary metrics ───────────────────────────────────────
stats = get_summary_stats()

c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Total Packets",   stats["total_packets"])
c2.metric("⚠️ Anomalies",       stats["total_anomalies"])
c3.metric("🔴 High Severity",   stats["high_severity"])
c4.metric("📋 Timeline Events", stats["total_events"])

st.markdown("---")


# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Live Traffic",
    "🚨 Anomaly Alerts",
    "📊 Analytics",
    "🕒 Incident Timeline",
    "🔬 Packet Inspector",
])


# ── Tab 1: Live traffic ───────────────────────────────────
with tab1:
    st.subheader("Recent Network Traffic")

    traffic = fetch_recent_traffic(50)
    if traffic:
        df = pd.DataFrame(traffic)

        # Colour anomalies
        def row_style(row):
            if row.get("is_anomaly"):
                return ['background-color: rgba(239,83,80,0.12)'] * len(row)
            return [''] * len(row)

        show_cols = ["timestamp", "src_ip", "dst_ip", "protocol",
                     "packet_size", "src_port", "dst_port", "ttl",
                     "is_anomaly", "anomaly_score"]
        existing = [c for c in show_cols if c in df.columns]
        st.dataframe(
            df[existing].style.apply(row_style, axis=1),
            use_container_width=True,
            height=420,
        )
    else:
        st.info("No traffic yet. Start the pipeline from the sidebar.")

    # Live log feed
    st.subheader("Live Log Feed")
    anomalies = fetch_recent_anomalies(10)
    recent    = fetch_recent_traffic(10)

    for pkt in recent[:10]:
        ts  = str(pkt.get("timestamp",""))[:19]
        src = pkt.get("src_ip","")
        dst = pkt.get("dst_ip","")
        if pkt.get("is_anomaly"):
            score = pkt.get("anomaly_score", 0)
            st.markdown(
                f'<div class="alert-anomaly">⚠ {ts} | {src} → {dst} | ANOMALY score={score:.3f}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="alert-normal">✓ {ts} | {src} → {dst} | NORMAL</div>',
                unsafe_allow_html=True,
            )


# ── Tab 2: Anomaly alerts ─────────────────────────────────
with tab2:
    st.subheader("🚨 Detected Anomalies")

    anomalies = fetch_recent_anomalies(100)
    if anomalies:
        df_a = pd.DataFrame(anomalies)

        severity_filter = st.multiselect(
            "Filter by severity",
            options=["HIGH", "MEDIUM", "LOW"],
            default=["HIGH", "MEDIUM", "LOW"],
        )
        df_filtered = df_a[df_a["severity"].isin(severity_filter)] if "severity" in df_a.columns else df_a

        def color_severity(val):
            if val == "HIGH":   return "background-color: rgba(198,40,40,0.3); color:#ef9a9a; font-weight:700"
            if val == "MEDIUM": return "background-color: rgba(245,127,23,0.3); color:#ffcc80; font-weight:700"
            return "background-color: rgba(46,125,50,0.3); color:#a5d6a7"

        show = ["timestamp","src_ip","dst_ip","protocol","anomaly_score","severity","description"]
        show = [c for c in show if c in df_filtered.columns]

        st.dataframe(
            df_filtered[show].style.applymap(color_severity, subset=["severity"]),
            use_container_width=True,
            height=480,
        )
        st.caption(f"Showing {len(df_filtered)} anomalies")
    else:
        st.info("No anomalies detected yet.")


# ── Tab 3: Analytics ──────────────────────────────────────
with tab3:
    st.subheader("📊 Traffic Analytics")

    traffic = fetch_recent_traffic(200)
    anomalies_all = fetch_recent_anomalies(200)

    col_left, col_right = st.columns(2)

    # Protocol distribution
    with col_left:
        if traffic:
            df_t = pd.DataFrame(traffic)
            if "protocol" in df_t.columns:
                proto_counts = df_t["protocol"].value_counts().reset_index()
                proto_counts.columns = ["Protocol", "Count"]
                fig = px.pie(
                    proto_counts, names="Protocol", values="Count",
                    title="Protocol Distribution",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                    hole=0.4,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")

    # Anomaly vs Normal
    with col_right:
        if traffic:
            df_t = pd.DataFrame(traffic)
            if "is_anomaly" in df_t.columns:
                counts = df_t["is_anomaly"].value_counts().rename({0:"Normal",1:"Anomaly"})
                fig2 = px.bar(
                    x=counts.index, y=counts.values,
                    title="Normal vs Anomaly Packets",
                    color=counts.index,
                    color_discrete_map={"Normal":"#42a5f5","Anomaly":"#ef5350"},
                    labels={"x":"Type","y":"Count"},
                )
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    showlegend=False,
                )
                st.plotly_chart(fig2, use_container_width=True)

    # Anomaly score over time
    if anomalies_all:
        df_a = pd.DataFrame(anomalies_all)
        if "timestamp" in df_a.columns and "anomaly_score" in df_a.columns:
            df_a["timestamp"] = pd.to_datetime(df_a["timestamp"], errors="coerce")
            df_a = df_a.dropna(subset=["timestamp"]).sort_values("timestamp")
            fig3 = px.line(
                df_a, x="timestamp", y="anomaly_score",
                title="Anomaly Score Over Time",
                color_discrete_sequence=["#ef5350"],
            )
            fig3.add_hline(y=0.75, line_dash="dash", line_color="#ff8a65",
                           annotation_text="HIGH threshold")
            fig3.add_hline(y=0.45, line_dash="dot",  line_color="#ffd54f",
                           annotation_text="MEDIUM threshold")
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig3, use_container_width=True)

    # Packet size distribution
    if traffic:
        df_t = pd.DataFrame(traffic)
        if "packet_size" in df_t.columns:
            fig4 = px.histogram(
                df_t, x="packet_size",
                title="Packet Size Distribution",
                nbins=40,
                color_discrete_sequence=["#42a5f5"],
            )
            fig4.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig4, use_container_width=True)


# ── Tab 4: Incident Timeline ──────────────────────────────
with tab4:
    st.subheader("🕒 Incident Timeline")

    col_rebuild, _ = st.columns([1,3])
    with col_rebuild:
        if st.button("🔄 Reconstruct Incidents"):
            incidents = reconstruct_incidents()
            st.success(f"Found {len(incidents)} incident(s).")

    events = fetch_incident_timeline()
    if events:
        df_ev = pd.DataFrame(events)

        # Scatter timeline chart
        if "timestamp" in df_ev.columns and "risk_score" in df_ev.columns:
            df_ev["timestamp"] = pd.to_datetime(df_ev["timestamp"], errors="coerce")
            fig5 = px.scatter(
                df_ev.dropna(subset=["timestamp"]),
                x="timestamp", y="risk_score",
                color="severity",
                hover_data=["incident_id","src_ip","dst_ip","event_type"],
                title="Incident Risk Score Timeline",
                color_discrete_map={"HIGH":"#ef5350","MEDIUM":"#ffa726","LOW":"#66bb6a"},
                size_max=14,
            )
            fig5.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig5, use_container_width=True)

        st.dataframe(df_ev, use_container_width=True, height=340)

        # Incident report download
        if "incident_id" in df_ev.columns:
            incident_ids = df_ev["incident_id"].dropna().unique().tolist()
            selected_inc = st.selectbox("Generate report for incident:", incident_ids)
            if selected_inc:
                report = generate_incident_report(selected_inc)
                st.code(report, language="text")
                st.download_button(
                    "📥 Download Report",
                    data=report,
                    file_name=f"{selected_inc}_report.txt",
                )
    else:
        st.info("No incident timeline events yet. Run the pipeline and click 'Reconstruct Incidents'.")


# ── Tab 5: Packet Inspector ───────────────────────────────
with tab5:
    st.subheader("🔬 Manual Packet Inspector")
    st.caption("Enter packet details to check whether the AI considers it anomalous.")

    with st.form("packet_form"):
        c1, c2 = st.columns(2)
        src_ip   = c1.text_input("Source IP",      "192.168.1.50")
        dst_ip   = c2.text_input("Destination IP", "10.0.0.5")

        c3, c4, c5 = st.columns(3)
        src_port = c3.number_input("Src Port", 0, 65535, 54321)
        dst_port = c4.number_input("Dst Port", 0, 65535, 22)
        protocol = c5.selectbox("Protocol", ["TCP", "UDP", "ICMP"])

        c6, c7, c8 = st.columns(3)
        pkt_size = c6.number_input("Packet Size (bytes)", 0, 65535, 1450)
        ttl      = c7.number_input("TTL", 0, 255, 25)
        flags    = c8.text_input("Flags (e.g. S, SA, RST)", "S")

        submitted = st.form_submit_button("🔍 Analyse Packet")

    if submitted:
        test_pkt = {
            "src_ip":      src_ip,
            "dst_ip":      dst_ip,
            "src_port":    int(src_port),
            "dst_port":    int(dst_port),
            "protocol":    protocol,
            "packet_size": int(pkt_size),
            "ttl":         int(ttl),
            "flags":       flags,
            "timestamp":   datetime.now().isoformat(),
        }
        result = predict_packet(test_pkt, model, scaler)

        if result["is_anomaly"]:
            st.error(
                f"⚠️ **ANOMALY DETECTED** | Severity: {result['severity']} | "
                f"Score: {result['anomaly_score']:.4f}\n\n{result['description']}"
            )
        else:
            st.success(
                f"✅ **NORMAL TRAFFIC** | Score: {result['anomaly_score']:.4f}\n\n"
                f"{result['description']}"
            )

        with st.expander("Feature Breakdown"):
            feat_df = pd.DataFrame.from_dict(
                result["features"], orient="index", columns=["Value"]
            )
            st.dataframe(feat_df, use_container_width=True)


# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.caption("🛡️ NetGrad | AI Network Anomaly Detection System")
