# Monte Carlo Covenant Engine

**Quantify the probability of debt covenant breach — not just pass/fail.**

A stochastic simulation engine that runs 10,000 Monte Carlo paths over 60-month horizons to model debt service coverage, leverage, and liquidity risk for acquisition financing scenarios.

Built to support real M&A financing decisions where point-estimate models aren't enough.

> All data in this repository is synthetic. Financial figures, entity names, and deal terms have been anonymized.

---

## Documents

- **[Analysis Memo (.docx)](docs/Monte%20Carlo%20Analysis%20Memo.docx)** — Full writeup of methodology, assumptions, and findings
- **[60-Month Model Output (.xlsx)](outputs/Monte%20Carlo%20Model%2060%20Months.xlsx)** — Formatted executive summary workbook with fan charts
- **[Raw Simulation Data (.xlsx)](outputs/mc_output_data.xlsx)** — All percentile distributions across 10,000 paths
- **[Simulation Engine (.py)](src/mc_covenants_v12_with_export.py)** — Source code (~2,300 lines)

---

## What It Does

**Simulation Core (10,000 paths x 60 months)**
- Stochastic revenue and expense generation via lognormal shocks with configurable volatility
- Regression-based forecasting with exponential recency weighting and seasonal decomposition
- Portfolio collections modeling using Dirichlet-jittered decay curves
- New client revenue overlay with lognormal noise and realization scaling
- Dual accounting basis support (GAAP vs. cash) for covenant vs. waterfall calculations

**Debt Mechanics**
- Two-tranche term debt with interest-only periods and independent amortization schedules
- Revolving line of credit with auto-draw/paydown logic, cash buffer maintenance, and annual rest compliance
- Full monthly cash waterfall: EBITDA -> debt service -> LOC paydown -> cash accumulation

**Covenant Testing**
- Borrower DSCR (TTM EBITDA / TTM total debt service)
- Global DSCR (includes personal cash flows and external debt service)
- Funded Debt-to-EBITDA (with step-down thresholds at month 12)
- Current Ratio (configurable treatment of restricted cash and RLOC in current liabilities)
- Compensating Balance (cash + eligible trust balances vs. percentage of funded debt)
- RLOC Rest (annual mandatory paydown period)

---

## Sample Results

From a 10,000-path simulation over 60 months:

- **Covenant breach probability:** 0% across all covenants at all checkpoints
- **Cash balance at month 24 (P50):** ~$3.4M | P5: ~$2.8M
- **TTM EBITDA at month 24 (P50):** ~$252K/month | P5: ~$168K/month
- **EBITDA negative probability:** ~18.9% (early-period volatility, resolves by month 18)

---

## Technical Highlights

- **No hardcoded values.** Every financial assumption flows from the CSV configuration layer, making it fully portable across deal structures.
- **Dual-basis accounting.** GAAP figures drive covenant calculations while cash-basis figures drive the cash waterfall — matching how covenants are actually tested vs. how cash actually moves.
- **Cure analysis.** For any non-zero breach probability, the engine calculates the minimum revenue increase or expense decrease required to eliminate breaches at P0, P5, P10, and P15 confidence levels.
- **Regression forecasting.** Revenue and expense projections use exponentially weighted seasonal decomposition with configurable half-life, rather than naive trailing averages.

---

## Tech Stack

Python, NumPy, Pandas, Matplotlib, openpyxl

---

## How to Run

```bash
pip install -r requirements.txt
cd src/
python mc_covenants_v12_with_export.py
```

All parameters are controlled via `data/assumptions.csv` — no hardcoded values. ~80 configurable inputs including volatility, covenant thresholds, debt terms, and forecast methodology.

---

## License

All Rights Reserved. See [LICENSE](LICENSE) for details. Shared for portfolio evaluation only.

---

*Landon Webster — Controller / Finance & Analytics*
