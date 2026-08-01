"""
app.py
------
AI Vendor Selection Assistant -- main Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ai_assistant import AIAssistant
from charts import (
    criteria_comparison_table,
    delivery_comparison_chart,
    price_comparison_chart,
    radar_chart,
    risk_heatmap,
    score_bar_chart,
)
from report import build_pdf_report
from scoring import ScoringEngine
from utils import (
    DEFAULT_WEIGHTS,
    SCORING_CRITERIA,
    get_logger,
    load_csv_bytes,
    load_uploaded_file,
    validate_dataframe,
)

logger = get_logger(__name__)
SAMPLE_DATA_PATH = Path(__file__).parent / "sample_data" / "sample_vendors.csv"

# --------------------------------------------------------------------------
# Page configuration & styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Vendor Selection Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #f5f7fa; }
    .stApp header { background-color: transparent; }
    h1, h2, h3 { color: #1a237e; }
    .metric-card {
        background: white; border-radius: 12px; padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 5px solid #1a73e8;
    }
    .recommend-banner {
        background: linear-gradient(90deg, #1a237e 0%, #1a73e8 100%);
        color: white; padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1rem;
    }
    .recommend-banner h2 { color: white; margin: 0; }
    .recommend-banner p { margin: 0.2rem 0 0 0; opacity: 0.92; }
    section[data-testid="stSidebar"] { background-color: #101935; }
    section[data-testid="stSidebar"] * { color: #eceff1 !important; }
    div[data-testid="stChatMessage"] { background: white; border-radius: 10px; }
    .stButton>button {
        background-color: #1a73e8; color: white; border-radius: 8px; border: none;
    }
    .stButton>button:hover { background-color: #1557b0; color: white; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

assistant = AIAssistant()

# --------------------------------------------------------------------------
# Sidebar -- data upload & weight configuration
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏢 AI Vendor Selection Assistant")
    st.caption("Enterprise procurement decision-support")

    st.markdown("### 📂 Vendor Data")
    uploaded_file = st.file_uploader("Upload vendor quotations", type=["csv", "xlsx", "xls"])
    use_sample = st.button("Load Sample Dataset", use_container_width=True)

    st.markdown("### ⚖️ Scoring Weights (%)")
    st.caption("Adjust priorities -- values auto-normalize to 100%.")

    if "weights" not in st.session_state:
        st.session_state.weights = dict(DEFAULT_WEIGHTS)

    weights_input = {}
    for criterion in SCORING_CRITERIA:
        weights_input[criterion] = st.slider(
            criterion, 0, 100, st.session_state.weights.get(criterion, DEFAULT_WEIGHTS[criterion]), 1
        )
    total_weight = sum(weights_input.values()) or 1
    st.session_state.weights = weights_input
    normalized_display = {k: round(v / total_weight * 100, 1) for k, v in weights_input.items()}

    if total_weight != 100:
        st.info(f"Raw total: {total_weight}%. Auto-normalized to 100% for scoring.")
    with st.expander("Normalized weights used"):
        for k, v in normalized_display.items():
            st.write(f"{k}: {v}%")

    if st.button("Reset to Default Weights", use_container_width=True):
        st.session_state.weights = dict(DEFAULT_WEIGHTS)
        st.rerun()

    st.markdown("---")
    st.caption("Built for AI Hackathon demo · Explainable, transparent scoring")

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
if use_sample:
    try:
        st.session_state.raw_df = load_csv_bytes(SAMPLE_DATA_PATH.read_bytes())
        st.session_state.chat_history = []
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load sample dataset: {exc}")

if uploaded_file is not None:
    try:
        df = load_uploaded_file(uploaded_file)
        st.session_state.raw_df = df
    except ValueError as exc:
        st.error(str(exc))

raw_df = st.session_state.raw_df

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🏢 AI Vendor Selection Assistant")
st.caption(
    "Compare vendor quotations, apply weighted scoring, and generate AI-backed "
    "business recommendations -- with full transparency into every score."
)

if raw_df is None:
    st.info(
        "👈 Upload a vendor quotation file (.csv or .xlsx) or click **Load Sample Dataset** "
        "in the sidebar to get started."
    )
    with st.expander("📋 Required columns"):
        from utils import REQUIRED_COLUMNS

        st.write(", ".join(REQUIRED_COLUMNS))
    st.stop()

# --------------------------------------------------------------------------
# Validate
# --------------------------------------------------------------------------
is_valid, errors = validate_dataframe(raw_df)
if not is_valid:
    st.error("The uploaded dataset has validation issues:")
    for e in errors:
        st.markdown(f"- {e}")
    st.stop()

# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------
try:
    engine = ScoringEngine(st.session_state.weights)
    scored_df = engine.compute_scores(raw_df)
except Exception as exc:  # noqa: BLE001
    logger.exception("Scoring failed")
    st.error(f"Scoring failed: {exc}")
    st.stop()

top_vendor = scored_df.iloc[0]

# --------------------------------------------------------------------------
# Recommendation banner
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="recommend-banner">
        <h2>✅ Recommended Vendor: {top_vendor['Vendor']}</h2>
        <p>Weighted Score: {top_vendor['Weighted Score']:.2f}/100 &nbsp;|&nbsp;
        Confidence: {engine.confidence_label(scored_df)} &nbsp;|&nbsp;
        Lead over runner-up: {engine.score_gap(scored_df):.2f} points</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_dashboard, tab_ai, tab_chat, tab_reports = st.tabs(
    ["📊 Dashboard", "🤖 AI Recommendation", "💬 Chat Assistant", "📄 Reports"]
)

# ===================== DASHBOARD =====================
with tab_dashboard:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><h4>Vendors Compared</h4>'
            f'<h2>{len(scored_df)}</h2></div>',
            unsafe_allow_html=True,
        )
    with c2:
        cheapest = scored_df.loc[scored_df["Total Price (INR)"].idxmin()]
        st.markdown(
            f'<div class="metric-card"><h4>Lowest Price</h4>'
            f'<h3>{cheapest["Vendor"]}</h3><p>₹{cheapest["Total Price (INR)"]:,.0f}</p></div>',
            unsafe_allow_html=True,
        )
    with c3:
        best_q = scored_df.loc[scored_df["Quality Score"].idxmax()]
        st.markdown(
            f'<div class="metric-card"><h4>Highest Quality</h4>'
            f'<h3>{best_q["Vendor"]}</h3><p>{best_q["Quality Score"]}/10</p></div>',
            unsafe_allow_html=True,
        )
    with c4:
        safest = scored_df.loc[scored_df["Vendor Risk Numeric"].idxmax()]
        st.markdown(
            f'<div class="metric-card"><h4>Lowest Risk</h4>'
            f'<h3>{safest["Vendor"]}</h3><p>{safest["Vendor Risk"]}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### ")
    col_a, col_b = st.columns([1, 1])
    fig_bar = score_bar_chart(scored_df)
    fig_radar = radar_chart(scored_df)
    fig_price = price_comparison_chart(scored_df)
    fig_delivery = delivery_comparison_chart(scored_df)
    fig_heatmap = risk_heatmap(scored_df)

    with col_a:
        st.plotly_chart(fig_bar, use_container_width=True)
        st.plotly_chart(fig_price, use_container_width=True)
    with col_b:
        st.plotly_chart(fig_radar, use_container_width=True)
        st.plotly_chart(fig_delivery, use_container_width=True)

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("### 📋 Criteria Comparison Table")
    st.dataframe(
        criteria_comparison_table(scored_df).style.background_gradient(
            cmap="RdYlGn", subset=SCORING_CRITERIA
        ),
        use_container_width=True,
        hide_index=True,
    )

# ===================== AI RECOMMENDATION =====================
with tab_ai:
    with st.spinner("Generating AI business reasoning..."):
        narrative = assistant.generate_recommendation(scored_df, st.session_state.weights, engine)

    st.subheader("📈 Executive Summary")
    st.write(narrative["executive_summary"])

    st.subheader("🎯 Why This Vendor Was Selected")
    st.write(narrative["selection_reasoning"])
    st.metric("Confidence Level", narrative["confidence"])

    st.subheader("💪 Strengths & Weaknesses by Vendor")
    st.markdown(narrative["strengths_weaknesses"])

    st.subheader("⚠️ Identified Risks")
    st.markdown(narrative["risks"])

    st.subheader("⚖️ Cost vs. Quality Trade-offs")
    st.write(narrative["trade_offs"])

    st.session_state["narrative"] = narrative

# ===================== CHAT ASSISTANT =====================
with tab_chat:
    st.subheader("💬 Ask the Procurement Assistant")
    st.caption(
        "Try: *Compare Vendor A and Vendor B* · *Why wasn't Vendor C selected?* · "
        "*Which vendor has the least risk?* · *Show only OEM-authorized vendors* · "
        "*What if delivery is the highest priority?* · *Generate an email summary for management*"
    )

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    user_query = st.chat_input("Ask a question about the vendors...")
    if user_query:
        st.session_state.chat_history.append(("user", user_query))
        with st.chat_message("user"):
            st.markdown(user_query)
        answer = assistant.answer_query(user_query, scored_df, st.session_state.weights, engine)
        st.session_state.chat_history.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)

    if st.session_state.chat_history and st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ===================== REPORTS =====================
with tab_reports:
    st.subheader("📄 Downloadable Procurement Report")
    st.write(
        "Generate a full PDF report including the procurement summary, vendor comparison "
        "table, charts, AI recommendation, and executive summary -- ready to share with management."
    )
    company_name = st.text_input("Company / Team name for report header", value="Procurement Team")

    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Building PDF report..."):
            narrative = st.session_state.get(
                "narrative", assistant.generate_recommendation(scored_df, st.session_state.weights, engine)
            )
            figures = {
                "score_bar": fig_bar,
                "radar": fig_radar,
                "price": fig_price,
                "delivery": fig_delivery,
                "heatmap": fig_heatmap,
            }
            try:
                pdf_bytes = build_pdf_report(
                    scored_df, narrative, normalized_display, figures, company_name=company_name
                )
                st.session_state["pdf_bytes"] = pdf_bytes
                st.success("Report generated successfully.")
            except Exception as exc:  # noqa: BLE001
                logger.exception("PDF generation failed")
                st.error(f"PDF generation failed: {exc}")

    if "pdf_bytes" in st.session_state:
        st.download_button(
            "⬇️ Download PDF Report",
            data=st.session_state["pdf_bytes"],
            file_name="vendor_selection_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("📤 Export Raw Comparison Data")
    csv_data = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Scored Data (CSV)",
        data=csv_data,
        file_name="vendor_scores.csv",
        mime="text/csv",
    )
