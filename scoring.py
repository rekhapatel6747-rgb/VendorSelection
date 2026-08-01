"""
scoring.py
----------
Weighted multi-criteria scoring engine for vendor selection.

The engine:
1. Converts categorical fields (Vendor Risk, Yes/No flags) into numeric form.
2. Min-max normalizes every scoring criterion onto a common 0-100 scale,
   inverting "lower is better" criteria so that 100 always means "best".
3. Computes a weighted composite score per vendor and ranks vendors.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from utils import (
    CRITERIA_SOURCE_COLUMN,
    LOWER_IS_BETTER,
    RISK_MAP,
    SCORING_CRITERIA,
    get_logger,
    risk_to_numeric,
    yes_no_to_numeric,
)

logger = get_logger(__name__)


class ScoringEngine:
    """Encapsulates normalization and weighted-score computation for vendors."""

    def __init__(self, weights: Dict[str, float]):
        """
        Parameters
        ----------
        weights: dict mapping each of utils.SCORING_CRITERIA to a weight
                 (percentages that should sum to ~100, but the engine
                 re-normalizes internally so any positive relative weights work).
        """
        missing = set(SCORING_CRITERIA) - set(weights)
        if missing:
            raise ValueError(f"Missing weight(s) for criteria: {missing}")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Sum of weights must be greater than zero.")
        # Store as fractions of 1.0 regardless of whether caller passed 0-100 or 0-1
        self.weights = {k: v / total for k, v in weights.items()}

    # ------------------------------------------------------------------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with categorical fields converted to numeric."""
        out = df.copy()
        out["Vendor Risk Numeric"] = risk_to_numeric(out["Vendor Risk"])
        for col in ("MSME Certified", "OEM Authorized", "Replacement Policy"):
            if col in out.columns:
                out[f"{col} Numeric"] = yes_no_to_numeric(out[col])
        return out

    # ------------------------------------------------------------------
    def _raw_value(self, df: pd.DataFrame, criterion: str) -> pd.Series:
        if criterion == "Vendor Risk":
            return df["Vendor Risk Numeric"]
        col = CRITERIA_SOURCE_COLUMN[criterion]
        return pd.to_numeric(df[col], errors="coerce")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a `<Criterion> Norm` column (0-100 scale, 100 = best) for every
        scoring criterion using min-max normalization.
        """
        out = self.prepare(df)
        for criterion in SCORING_CRITERIA:
            raw = self._raw_value(out, criterion)
            lo, hi = raw.min(), raw.max()
            if pd.isna(lo) or pd.isna(hi) or hi == lo:
                # All vendors tie on this criterion -> full marks to all
                norm = pd.Series(100.0, index=raw.index)
            else:
                if criterion in LOWER_IS_BETTER:
                    norm = (hi - raw) / (hi - lo) * 100
                else:
                    norm = (raw - lo) / (hi - lo) * 100
            out[f"{criterion} Norm"] = norm.round(2)
        return out

    def compute_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full pipeline: normalize criteria, apply weights, compute the
        composite Weighted Score, and rank vendors (1 = best).
        """
        out = self.normalize(df)
        out["Weighted Score"] = 0.0
        for criterion in SCORING_CRITERIA:
            out["Weighted Score"] += out[f"{criterion} Norm"] * self.weights[criterion]
        out["Weighted Score"] = out["Weighted Score"].round(2)
        out = out.sort_values("Weighted Score", ascending=False).reset_index(drop=True)
        out["Rank"] = out.index + 1
        logger.info(
            "Scored %d vendors; top vendor = %s (%.2f)",
            len(out),
            out.iloc[0]["Vendor"],
            out.iloc[0]["Weighted Score"],
        )
        return out

    # ------------------------------------------------------------------
    def score_gap(self, scored_df: pd.DataFrame) -> float:
        """Point gap between the #1 and #2 ranked vendor (confidence proxy)."""
        if len(scored_df) < 2:
            return 100.0
        s = scored_df.sort_values("Weighted Score", ascending=False)
        return round(float(s.iloc[0]["Weighted Score"] - s.iloc[1]["Weighted Score"]), 2)

    def confidence_label(self, scored_df: pd.DataFrame) -> str:
        gap = self.score_gap(scored_df)
        if gap >= 10:
            return "High"
        if gap >= 4:
            return "Medium"
        return "Low"
