# 🏢 AI Vendor Selection Assistant

An enterprise-grade procurement decision-support tool built with **Python + Streamlit**.
It helps procurement teams compare multiple vendor quotations and recommends the best
vendor using **transparent weighted scoring** plus an **AI-generated business narrative**
(strengths/weaknesses, risks, trade-offs, confidence, executive summary).

Built for demonstrating explainability, transparency, and business value in an
AI Hackathon setting — not just a black-box score.

---

## ✨ Features

- 📂 Upload vendor quotations as **CSV or Excel (.xlsx)**, with schema validation and
  friendly error messages.
- ⚖️ **Dynamic weighted scoring** across 8 criteria (Price, Quality, Warranty, Vendor Risk,
  Delivery, Past Performance, Compliance, Support SLA) with sidebar sliders that
  auto-normalize to 100%.
- 🤖 **AI recommendation engine** that explains *why* a vendor was chosen, strengths/
  weaknesses per vendor, identified risks, cost-vs-quality trade-offs, a confidence
  level, and an executive summary. Works fully offline (rule-based reasoning) and can
  optionally use the Anthropic API to polish prose if `ANTHROPIC_API_KEY` is set.
- 💬 **Chat assistant** for natural questions: *"Compare Vendor A and Vendor B"*,
  *"Why wasn't Vendor X selected?"*, *"Which vendor has the least risk?"*,
  *"Show only OEM-authorized vendors"*, *"What if delivery is the highest priority?"*,
  *"Generate an email summary for management"*, and more.
- 📊 **Visual analytics**: score ranking bar chart, radar/spider comparison, price
  comparison, delivery comparison, criteria heatmap, and a full comparison table.
- 📄 **Downloadable PDF report** with procurement summary, comparison table, charts,
  AI recommendation, and executive summary — ready to share with management.
- 🧪 Bundled **realistic sample dataset** (laptop procurement, 6 vendors) so the app
  runs immediately with zero setup.

---

## 🗂️ Project Structure

```
AI-Vendor-Selection-Assistant/
├── app.py                     # Streamlit UI & orchestration
├── scoring.py                 # ScoringEngine: normalization + weighted scoring
├── ai_assistant.py            # AI narrative generation + chat query handling
├── charts.py                  # Plotly chart builders
├── report.py                  # PDF report generation (reportlab)
├── utils.py                   # Constants, logging, validation, file loading
├── sample_data/
│   ├── sample_vendors.csv
│   └── sample_vendors.xlsx
├── requirements.txt
└── README.md
```

Clean separation of concerns: **data/validation** (`utils.py`) → **scoring**
(`scoring.py`) → **AI reasoning** (`ai_assistant.py`) → **visualization**
(`charts.py`) → **reporting** (`report.py`) → **UI** (`app.py`).

---

## 🚀 Getting Started

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/VendorSelection.git
cd VendorSelection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Enable AI-polished narratives

The app generates fully explainable, rule-based business narratives out of the box.
To optionally have Claude polish the executive summary into more natural prose, set:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # Windows: set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`), then click
**"Load Sample Dataset"** in the sidebar to try it immediately, or upload your own
vendor quotation file.

---

## 📋 Expected Dataset Columns

| Column | Type | Notes |
|---|---|---|
| Vendor | text | Unique vendor name |
| Unit Price (INR) | number | |
| Quantity | number | |
| Total Price (INR) | number | Lower is better |
| Warranty (Years) | number | Higher is better |
| Delivery (Days) | number | Lower is better |
| Quality Score | number | Higher is better |
| Vendor Risk | Low / Medium / High | Converted to 100 / 60 / 20 |
| Payment Terms (Days) | number | Informational |
| Past Performance (/5) | number | Higher is better |
| Support SLA (Hours) | number | Lower is better (faster response) |
| Replacement Policy | Yes / No | |
| Compliance Score | number | Higher is better |
| MSME Certified | Yes / No | |
| OEM Authorized | Yes / No | |
| Location | text | Informational |

## ⚖️ Default Weightage

| Criterion | Weight |
|---|---|
| Price | 30% |
| Quality | 20% |
| Warranty | 15% |
| Vendor Risk | 10% |
| Delivery | 10% |
| Past Performance | 5% |
| Compliance | 5% |
| Support SLA | 5% |

All weights are adjustable live via sidebar sliders and are automatically
re-normalized to sum to 100% when scoring.

---

## 🧠 How Scoring Works

1. **Categorical conversion**: `Vendor Risk` (Low/Medium/High → 100/60/20),
   Yes/No fields → 1/0.
2. **Min-max normalization**: every criterion is scaled to 0-100 across the
   uploaded vendors. "Lower is better" criteria (Price, Delivery, Support SLA)
   are inverted so 100 always means "best".
3. **Weighted composite score**: `Σ (normalized_criterion × weight)`.
4. **Ranking**: vendors sorted descending by composite score; #1 is recommended.
5. **Confidence**: derived from the point gap between the #1 and #2 vendor
   (High ≥10 pts, Medium ≥4 pts, Low otherwise).

---

## 🛠️ Tech Stack

- **Streamlit** — UI framework
- **pandas / numpy** — data processing
- **Plotly** — interactive charts
- **ReportLab + Kaleido** — PDF report generation with embedded chart images
- **Anthropic SDK** (optional) — AI-polished executive summaries

---

## 📦 Deploying

Works out of the box on **Streamlit Community Cloud**: point it at `app.py` in this
repo. No secrets are required unless you want the optional LLM polish step, in which
case add `ANTHROPIC_API_KEY` under app secrets.

---

## 📝 License

MIT — free to use and adapt for your organization or hackathon.
