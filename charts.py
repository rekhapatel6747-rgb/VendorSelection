"""
charts.py
---------
Plotly chart builders for the AI Vendor Selection Assistant dashboard.
Every function accepts the scored dataframe (output of ScoringEngine.compute_scores)
and returns a plotly.graph_objects.Figure ready for st.plotly_chart().
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import SCORING_CRITERIA

TEMPLATE = "plotly_white"
COLOR_SEQUENCE = px.colors.qualitative.Bold


def score_bar_chart(scored_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of the composite weighted score, best vendor on top."""
    df = scored_df.sort_values("Weighted Score", ascending=True)
    colors = [
        "#1a73e8" if v == scored_df.iloc[0]["Vendor"] else "#90a4ae" for v in df["Vendor"]
    ]
    fig = go.Figure(
        go.Bar(
            x=df["Weighted Score"],
            y=df["Vendor"],
            orientation="h",
            marker_color=colors,
            text=df["Weighted Score"].map(lambda v: f"{v:.1f}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Vendor Weighted Score Ranking",
        xaxis_title="Weighted Score (0-100)",
        yaxis_title="",
        template=TEMPLATE,
        height=420,
        margin=dict(l=10, r=40, t=50, b=10),
    )
    return fig


def radar_chart(scored_df: pd.DataFrame, vendors: list[str] | None = None) -> go.Figure:
    """Radar/spider chart comparing normalized criteria scores across vendors."""
    if vendors:
        df = scored_df[scored_df["Vendor"].isin(vendors)]
    else:
        df = scored_df

    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[f"{c} Norm"] for c in SCORING_CRITERIA]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=SCORING_CRITERIA + [SCORING_CRITERIA[0]],
                fill="toself",
                name=row["Vendor"],
                line_color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)],
                opacity=0.75,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Multi-Criteria Comparison (Normalized 0-100)",
        template=TEMPLATE,
        height=480,
        showlegend=True,
    )
    return fig


def price_comparison_chart(scored_df: pd.DataFrame) -> go.Figure:
    df = scored_df.sort_values("Total Price (INR)")
    fig = px.bar(
        df,
        x="Vendor",
        y="Total Price (INR)",
        color="Vendor",
        color_discrete_sequence=COLOR_SEQUENCE,
        text_auto=".2s",
        title="Total Price Comparison",
    )
    fig.update_layout(template=TEMPLATE, height=380, showlegend=False, margin=dict(t=50, b=10))
    return fig


def delivery_comparison_chart(scored_df: pd.DataFrame) -> go.Figure:
    df = scored_df.sort_values("Delivery (Days)")
    fig = px.bar(
        df,
        x="Vendor",
        y="Delivery (Days)",
        color="Vendor",
        color_discrete_sequence=COLOR_SEQUENCE,
        text_auto=True,
        title="Delivery Timeline Comparison (Days)",
    )
    fig.update_layout(template=TEMPLATE, height=380, showlegend=False, margin=dict(t=50, b=10))
    return fig


def risk_heatmap(scored_df: pd.DataFrame) -> go.Figure:
    """Heatmap of normalized criteria scores per vendor -- doubles as a risk/quality matrix."""
    matrix = scored_df.set_index("Vendor")[[f"{c} Norm" for c in SCORING_CRITERIA]]
    matrix.columns = SCORING_CRITERIA
    fig = px.imshow(
        matrix,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        text_auto=".0f",
        labels=dict(x="Criteria", y="Vendor", color="Score"),
        title="Vendor Criteria Heatmap (Green = Strong, Red = Weak)",
    )
    fig.update_layout(template=TEMPLATE, height=max(320, 60 * len(scored_df)), margin=dict(t=50, b=10))
    return fig


def criteria_comparison_table(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted dataframe (for st.dataframe) of raw + normalized criteria."""
    cols = ["Rank", "Vendor", "Weighted Score"] + [f"{c} Norm" for c in SCORING_CRITERIA]
    table = scored_df[cols].copy()
    table.columns = ["Rank", "Vendor", "Weighted Score"] + SCORING_CRITERIA
    return table.sort_values("Rank")
