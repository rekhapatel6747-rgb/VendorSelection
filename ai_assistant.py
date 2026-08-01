"""
ai_assistant.py
----------------
Business-reasoning layer for the AI Vendor Selection Assistant.

Two capabilities are provided:

1. `AIAssistant.generate_recommendation()` builds a structured, explainable
   narrative (executive summary, strengths/weaknesses, risks, trade-offs,
   confidence) directly from the scored data using transparent business
   rules -- so the app works fully offline, out of the box.

2. `AIAssistant.answer_query()` is a lightweight natural-language chat
   interface that recognizes common procurement questions (comparisons,
   "why not vendor X", filters, what-if re-weighting, email drafting) and
   answers them using the same scored data.

If an `ANTHROPIC_API_KEY` is available in the environment, the assistant
will optionally use it to polish the narrative into more natural prose via
the Anthropic Messages API. This is entirely optional -- everything works
without it.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import pandas as pd

from scoring import ScoringEngine
from utils import SCORING_CRITERIA, get_logger

logger = get_logger(__name__)


class AIAssistant:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm and bool(os.environ.get("ANTHROPIC_API_KEY"))
        if self.use_llm:
            try:
                import anthropic  # noqa: F401

                self._client = __import__("anthropic").Anthropic()
            except Exception:  # noqa: BLE001
                logger.warning("Anthropic SDK not available; using rule-based narrative only.")
                self.use_llm = False

    # ------------------------------------------------------------------
    # Structured explainable narrative
    # ------------------------------------------------------------------
    def generate_recommendation(
        self, scored_df: pd.DataFrame, weights: Dict[str, float], engine: ScoringEngine
    ) -> Dict[str, str]:
        """
        Returns a dict with keys:
        executive_summary, selection_reasoning, strengths_weaknesses,
        risks, trade_offs, confidence
        """
        top = scored_df.iloc[0]
        runner_up = scored_df.iloc[1] if len(scored_df) > 1 else None
        gap = engine.score_gap(scored_df)
        confidence = engine.confidence_label(scored_df)

        # --- Why the top vendor was chosen: its 3 strongest criteria ---
        norm_cols = {c: top[f"{c} Norm"] for c in SCORING_CRITERIA}
        top_criteria = sorted(norm_cols.items(), key=lambda kv: kv[1], reverse=True)[:3]
        weak_criteria = sorted(norm_cols.items(), key=lambda kv: kv[1])[:2]

        reasoning_lines = [
            f"**{top['Vendor']}** achieved the highest weighted score of "
            f"**{top['Weighted Score']:.2f}/100**, driven primarily by strong performance in "
            + ", ".join(f"{name} ({score:.0f}/100)" for name, score in top_criteria)
            + "."
        ]
        if runner_up is not None:
            reasoning_lines.append(
                f"It outperformed the runner-up, {runner_up['Vendor']} "
                f"({runner_up['Weighted Score']:.2f}/100), by a margin of {gap:.2f} points "
                f"under the current weight configuration."
            )
        if weak_criteria:
            reasoning_lines.append(
                "Relative weak points to monitor: "
                + ", ".join(f"{name} ({score:.0f}/100)" for name, score in weak_criteria)
                + "."
            )
        selection_reasoning = " ".join(reasoning_lines)

        # --- Strengths & weaknesses per vendor ---
        sw_lines: List[str] = []
        for _, row in scored_df.iterrows():
            strengths = sorted(
                ((c, row[f"{c} Norm"]) for c in SCORING_CRITERIA), key=lambda kv: kv[1], reverse=True
            )[:2]
            weaknesses = sorted(
                ((c, row[f"{c} Norm"]) for c in SCORING_CRITERIA), key=lambda kv: kv[1]
            )[:2]
            sw_lines.append(
                f"- **{row['Vendor']}** (Rank {int(row['Rank'])}, Score {row['Weighted Score']:.2f}) — "
                f"Strengths: {', '.join(f'{n} ({v:.0f})' for n, v in strengths)}. "
                f"Weaknesses: {', '.join(f'{n} ({v:.0f})' for n, v in weaknesses)}."
            )
        strengths_weaknesses = "\n".join(sw_lines)

        # --- Risks ---
        risk_lines: List[str] = []
        high_risk = scored_df[scored_df["Vendor Risk Numeric"] <= 20]
        if not high_risk.empty:
            risk_lines.append(
                "High risk exposure identified for: "
                + ", ".join(high_risk["Vendor"].tolist())
                + ". Recommend contractual safeguards (penalty clauses, escrow, phased payments)."
            )
        low_compliance = scored_df[scored_df["Compliance Score"] < scored_df["Compliance Score"].median()]
        if not low_compliance.empty:
            risk_lines.append(
                "Below-median compliance scores for: "
                + ", ".join(low_compliance["Vendor"].tolist())
                + ". Suggest a compliance audit before final sign-off."
            )
        slow_delivery = scored_df.sort_values("Delivery (Days)", ascending=False).iloc[0]
        risk_lines.append(
            f"Longest lead time is {slow_delivery['Vendor']} at {slow_delivery['Delivery (Days)']} days "
            "-- factor this into project scheduling if selected."
        )
        if not risk_lines:
            risk_lines.append("No material risk flags identified across the vendor pool.")
        risks = "\n".join(f"- {line}" for line in risk_lines)

        # --- Trade-offs: cheapest vs highest quality ---
        cheapest = scored_df.loc[scored_df["Total Price (INR)"].idxmin()]
        best_quality = scored_df.loc[scored_df["Quality Score"].idxmax()]
        if cheapest["Vendor"] == best_quality["Vendor"]:
            trade_offs = (
                f"**{cheapest['Vendor']}** offers both the lowest total price and the highest "
                "quality score in this pool, so there is no cost-quality trade-off to weigh here."
            )
        else:
            price_diff = best_quality["Total Price (INR)"] - cheapest["Total Price (INR)"]
            trade_offs = (
                f"**{cheapest['Vendor']}** is the lowest-cost option "
                f"(₹{cheapest['Total Price (INR)']:,.0f}), while **{best_quality['Vendor']}** leads on "
                f"quality ({best_quality['Quality Score']}/10). Choosing quality over cost here carries "
                f"a premium of approximately ₹{price_diff:,.0f}. The recommended vendor "
                f"({top['Vendor']}) balances both dimensions under the selected weighting."
            )

        # --- Executive summary ---
        executive_summary = (
            f"Based on a weighted evaluation across {len(SCORING_CRITERIA)} procurement criteria "
            f"(price, quality, warranty, risk, delivery, past performance, compliance, and support SLA), "
            f"**{top['Vendor']}** is the recommended vendor with a composite score of "
            f"{top['Weighted Score']:.2f}/100 and **{confidence.lower()} confidence** "
            f"(a {gap:.2f}-point lead over the next-best option). "
            f"The decision reflects the procurement team's current priority weighting; adjusting "
            f"weights (e.g., prioritizing delivery speed or risk) may change the outcome and can be "
            f"tested live in this tool."
        )

        result = {
            "executive_summary": executive_summary,
            "selection_reasoning": selection_reasoning,
            "strengths_weaknesses": strengths_weaknesses,
            "risks": risks,
            "trade_offs": trade_offs,
            "confidence": confidence,
            "score_gap": gap,
        }

        if self.use_llm:
            result = self._polish_with_llm(result, top["Vendor"])
        return result

    # ------------------------------------------------------------------
    def _polish_with_llm(self, sections: Dict[str, str], top_vendor: str) -> Dict[str, str]:
        """Optionally rewrite the executive summary in more natural prose via Claude."""
        try:
            prompt = (
                "Rewrite the following procurement executive summary in confident, "
                "concise business English (max 120 words). Keep every factual claim "
                "unchanged, do not invent numbers.\n\n" + sections["executive_summary"]
            )
            msg = self._client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
            if text.strip():
                sections["executive_summary"] = text.strip()
        except Exception:  # noqa: BLE001
            logger.warning("LLM polish step failed; falling back to rule-based summary.", exc_info=True)
        return sections

    # ------------------------------------------------------------------
    # Chat assistant
    # ------------------------------------------------------------------
    def answer_query(
        self,
        query: str,
        scored_df: pd.DataFrame,
        weights: Dict[str, float],
        engine: ScoringEngine,
    ) -> str:
        """Route a free-text question to the appropriate rule-based handler."""
        q = query.strip().lower()
        vendors = {v.lower(): v for v in scored_df["Vendor"]}

        def find_vendor(text: str) -> Optional[str]:
            for lower_name, real_name in vendors.items():
                if lower_name in text:
                    return real_name
            return None

        # "compare A and B"
        compare_match = re.search(r"compare (.+?) (?:and|with|vs\.?) (.+)", q)
        if compare_match:
            v1 = find_vendor(compare_match.group(1))
            v2 = find_vendor(compare_match.group(2))
            if v1 and v2:
                return self._compare_vendors(scored_df, v1, v2)

        # "why wasn't X selected" / "why not X"
        if "why" in q and ("not" in q or "wasn't" in q or "isn't" in q or "didn't" in q):
            vendor = find_vendor(q)
            if vendor:
                return self._why_not_selected(scored_df, vendor)

        # "least risk" / "lowest risk" / "safest"
        if "least risk" in q or "lowest risk" in q or "safest" in q:
            row = scored_df.loc[scored_df["Vendor Risk Numeric"].idxmax()]
            return (
                f"**{row['Vendor']}** carries the least risk, rated '{row['Vendor Risk']}' "
                f"(risk score {row['Vendor Risk Numeric']}/100)."
            )

        # "highest risk"
        if "highest risk" in q or "most risk" in q or "riskiest" in q:
            row = scored_df.loc[scored_df["Vendor Risk Numeric"].idxmin()]
            return (
                f"**{row['Vendor']}** carries the highest risk, rated '{row['Vendor Risk']}' "
                f"(risk score {row['Vendor Risk Numeric']}/100). Recommend additional due diligence."
            )

        # "OEM authorized" filter
        if "oem" in q:
            subset = scored_df[scored_df["OEM Authorized"].astype(str).str.title() == "Yes"]
            if subset.empty:
                return "No vendors in this dataset are OEM authorized."
            return "OEM-authorized vendors: " + ", ".join(
                f"{v} (Rank {r})" for v, r in zip(subset["Vendor"], subset["Rank"])
            )

        # "MSME" filter
        if "msme" in q:
            subset = scored_df[scored_df["MSME Certified"].astype(str).str.title() == "Yes"]
            if subset.empty:
                return "No vendors in this dataset are MSME certified."
            return "MSME-certified vendors: " + ", ".join(
                f"{v} (Rank {r})" for v, r in zip(subset["Vendor"], subset["Rank"])
            )

        # "best warranty"
        if "warranty" in q and ("best" in q or "longest" in q or "highest" in q):
            row = scored_df.loc[scored_df["Warranty (Years)"].idxmax()]
            return f"**{row['Vendor']}** offers the best warranty at {row['Warranty (Years)']} years."

        # "cheapest" / "lowest price"
        if "cheapest" in q or "lowest price" in q or "lowest cost" in q:
            row = scored_df.loc[scored_df["Total Price (INR)"].idxmin()]
            return f"**{row['Vendor']}** is the cheapest at ₹{row['Total Price (INR)']:,.0f}."

        # "fastest delivery"
        if "fastest" in q or ("delivery" in q and ("best" in q or "quick" in q or "shortest" in q)):
            row = scored_df.loc[scored_df["Delivery (Days)"].idxmin()]
            return f"**{row['Vendor']}** has the fastest delivery at {row['Delivery (Days)']} days."

        # "what if delivery is the highest priority" -> what-if re-weighting
        what_if = re.search(r"what if (.+?) (?:is|was) (?:the )?(?:highest|top|most important)", q)
        if what_if:
            criterion = self._match_criterion(what_if.group(1))
            if criterion:
                return self._what_if_priority(scored_df, weights, criterion, engine)

        # email summary
        if "email" in q and ("summary" in q or "management" in q or "draft" in q):
            return self._draft_email(scored_df)

        # top vendor / recommendation
        if "recommend" in q or "best vendor" in q or "top vendor" in q or "who should we" in q:
            top = scored_df.iloc[0]
            return (
                f"**{top['Vendor']}** is the recommended vendor with a weighted score of "
                f"{top['Weighted Score']:.2f}/100 (Rank 1)."
            )

        # Single vendor lookup
        vendor = find_vendor(q)
        if vendor:
            return self._vendor_profile(scored_df, vendor)

        return (
            "I can help with vendor comparisons, risk analysis, filters (OEM/MSME), "
            "warranty/price/delivery lookups, what-if re-weighting, or drafting an email "
            "summary. Try: *'Compare Vendor A and Vendor B'* or *'Why wasn't Vendor C selected?'*"
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _match_criterion(text: str) -> Optional[str]:
        text = text.lower()
        for c in SCORING_CRITERIA:
            if c.lower() in text:
                return c
        aliases = {
            "cost": "Price", "price": "Price", "risk": "Vendor Risk",
            "sla": "Support SLA", "support": "Support SLA",
            "performance": "Past Performance", "warranty": "Warranty",
            "compliance": "Compliance", "quality": "Quality", "delivery": "Delivery",
        }
        for key, val in aliases.items():
            if key in text:
                return val
        return None

    @staticmethod
    def _compare_vendors(scored_df: pd.DataFrame, v1: str, v2: str) -> str:
        r1 = scored_df[scored_df["Vendor"] == v1].iloc[0]
        r2 = scored_df[scored_df["Vendor"] == v2].iloc[0]
        lines = [f"**{v1}** (Rank {int(r1['Rank'])}, {r1['Weighted Score']:.2f}) vs "
                 f"**{v2}** (Rank {int(r2['Rank'])}, {r2['Weighted Score']:.2f}):"]
        for c in SCORING_CRITERIA:
            a, b = r1[f"{c} Norm"], r2[f"{c} Norm"]
            better = v1 if a > b else (v2 if b > a else "Tie")
            lines.append(f"- {c}: {v1}={a:.0f} vs {v2}={b:.0f} → **{better}** leads")
        winner = v1 if r1["Weighted Score"] > r2["Weighted Score"] else v2
        lines.append(f"\nOverall, **{winner}** has the higher composite score under current weights.")
        return "\n".join(lines)

    @staticmethod
    def _why_not_selected(scored_df: pd.DataFrame, vendor: str) -> str:
        top = scored_df.iloc[0]
        row = scored_df[scored_df["Vendor"] == vendor].iloc[0]
        if vendor == top["Vendor"]:
            return f"**{vendor}** *is* the currently recommended vendor (Rank 1)."
        gap = top["Weighted Score"] - row["Weighted Score"]
        weak = sorted(
            ((c, row[f"{c} Norm"]) for c in SCORING_CRITERIA), key=lambda kv: kv[1]
        )[:3]
        return (
            f"**{vendor}** ranked #{int(row['Rank'])} with a score of {row['Weighted Score']:.2f}, "
            f"trailing the recommended vendor **{top['Vendor']}** by {gap:.2f} points. "
            f"Its weakest areas were: {', '.join(f'{n} ({v:.0f}/100)' for n, v in weak)}."
        )

    @staticmethod
    def _vendor_profile(scored_df: pd.DataFrame, vendor: str) -> str:
        row = scored_df[scored_df["Vendor"] == vendor].iloc[0]
        return (
            f"**{vendor}** — Rank {int(row['Rank'])}, Weighted Score {row['Weighted Score']:.2f}/100. "
            f"Price ₹{row['Total Price (INR)']:,.0f}, Warranty {row['Warranty (Years)']} yrs, "
            f"Delivery {row['Delivery (Days)']} days, Risk {row['Vendor Risk']}, "
            f"Quality {row['Quality Score']}/10, Compliance {row['Compliance Score']}."
        )

    @staticmethod
    def _what_if_priority(
        scored_df: pd.DataFrame, weights: Dict[str, float], criterion: str, engine: ScoringEngine
    ) -> str:
        new_weights = {k: 5 for k in weights}
        new_weights[criterion] = 65
        new_engine = ScoringEngine(new_weights)
        base_df = scored_df.drop(
            columns=[c for c in scored_df.columns if c.endswith("Norm") or c == "Weighted Score" or c == "Rank"]
        )
        re_scored = new_engine.compute_scores(base_df)
        new_top = re_scored.iloc[0]
        old_top = scored_df.iloc[0]
        if new_top["Vendor"] == old_top["Vendor"]:
            return (
                f"If **{criterion}** became the top priority, **{new_top['Vendor']}** would still be "
                f"the recommended vendor (score {new_top['Weighted Score']:.2f}/100) -- the recommendation is robust to this change."
            )
        return (
            f"If **{criterion}** became the top priority, the recommendation would shift from "
            f"**{old_top['Vendor']}** to **{new_top['Vendor']}** (new score {new_top['Weighted Score']:.2f}/100). "
            "Use the sidebar sliders to apply this change permanently."
        )

    @staticmethod
    def _draft_email(scored_df: pd.DataFrame) -> str:
        top = scored_df.iloc[0]
        others = scored_df.iloc[1:4]
        lines = [
            "**Subject: Vendor Selection Recommendation — Procurement Decision**",
            "",
            "Hi team,",
            "",
            f"Following a weighted evaluation of {len(scored_df)} vendor proposals across price, "
            "quality, warranty, risk, delivery, past performance, compliance, and support SLA, we "
            f"recommend proceeding with **{top['Vendor']}** (composite score {top['Weighted Score']:.2f}/100).",
            "",
            "Other vendors considered:",
        ]
        for _, r in others.iterrows():
            lines.append(f"- {r['Vendor']}: {r['Weighted Score']:.2f}/100 (Rank {int(r['Rank'])})")
        lines += [
            "",
            "Full scoring breakdown and supporting charts are attached in the procurement report.",
            "",
            "Best regards,",
            "Procurement Team",
        ]
        return "\n".join(lines)
