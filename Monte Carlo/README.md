# Monte Carlo Covenant Engine

A stochastic simulation engine for modeling debt covenant compliance risk over multi-year horizons. Built for acquisition financing scenarios where a borrower must maintain compliance with multiple financial covenants simultaneously under uncertain revenue and expense conditions.

> ⚠️ **All data in this repository is synthetic.** Financial figures, entity names, and deal terms have been anonymized. No real company, client, or account data is present.

---

## Problem Statement

When structuring acquisition debt with multiple covenant tests — DSCR, Debt-to-EBITDA, current ratio, compensating balance, RLOC rest — point-estimate models give a single "pass/fail" answer that tells you nothing about tail risk. A Monte Carlo approach quantifies the *probability* of breach at each covenant checkpoint, identifies which covenants bind first, and measures how much revenue improvement or expense reduction would be required to cure breaches at various confidence levels.

This engine was built to support real M&A financing decisions where the borrower needed to understand covenant headroom across 10,000 simulated paths over a 60-month horizon.

## What It Does

**Simulation Core (10,000 paths × 60 months)**
- Stochastic revenue and expense generation via lognormal shocks with configurable volatility
- Regression-based forecasting with exponential recency weighting and seasonal decomposition
- Portfolio collections modeling using Dirichlet-jittered decay curves
- New client revenue overlay with lognormal noise and realization scaling
- Dual accounting basis support (GAAP vs. cash) for covenant vs. waterfall calculations

**Debt Mechanics**
- Two-tranche term debt with interest-only periods and independent amortization schedules
- Revolving line of credit with auto-draw/paydown logic, cash buffer maintenance, and annual rest compliance
- Full monthly cash waterfall: EBITDA → debt service → LOC paydown → cash accumulation

**Covenant Testing**
- Borrower DSCR (TTM EBITDA / TTM total debt service)
- Global DSCR (includes personal cash flows and external debt service)
- Funded Debt-to-EBITDA (with step-down thresholds at month 12)
- Current Ratio (configurable treatment of restricted cash and RLOC in current liabilities)
- Compensating Balance (cash + eligible trust balances vs. percentage of funded debt)
- RLOC Rest (annual mandatory paydown period)

**Outputs**
- Cumulative and point-in-time breach probabilities at 12/18/24/36/48/60 month checkpoints
- Percentile distributions (P5/P10/P25/P50/P75/P90/P95) for cash, EBITDA, and debt balances
- Cure analysis: required revenue increase or expense decrease to eliminate breaches at specified confidence levels
- Full Excel export with executive summary, fan charts, and assumption documentation

## Repository Structure

```
monte-carlo-covenant-engine/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   └── mc_covenants_v12_with_export.py    # Simulation engine (~2,300 lines)
├── data/
│   ├── assumptions.csv                     # All configurable parameters
│   ├── RevAndExp.csv                       # Historical revenue & expense series
│   ├── curve.csv                           # Portfolio collections decay curve
│   ├── new_client_revenue.csv              # Projected new client revenue overlay
│   └── cash_basis_baseline.csv             # Cash-basis baseline for split accounting
├── outputs/
│   ├── mc_output_data.xlsx                 # Raw simulation output (all percentiles)
│   └── Monte_Carlo_Model_60_Months.xlsx    # Formatted executive summary workbook
└── docs/
    └── methodology.md                      # (planned) Technical methodology writeup
```

## How to Run

```bash
pip install numpy pandas matplotlib openpyxl
cd src/
python mc_covenants_v12_with_export.py
```

The engine reads from `../data/` by default. All parameters are controlled via `assumptions.csv` — no hardcoded values in the simulation logic.

### Key Configuration Parameters

| Parameter | Description | Default |
|---|---|---|
| `n_sims` | Number of Monte Carlo paths | 10,000 |
| `horizon_months` | Simulation horizon | 60 |
| `forecast_mode` | `regression` (trend + seasonal) or `static` (repeat last 12mo) | `regression` |
| `revenue_sigma` | Revenue volatility (lognormal CV) | 0.08 |
| `expenses_sigma` | Expense volatility (lognormal CV) | 0.06 |
| `dscr_min` | Borrower DSCR covenant minimum | 1.30 |
| `debt_to_ebitda_max_first12` | D/EBITDA max (months 1–12) | 2.25 |
| `debt_to_ebitda_max_after12` | D/EBITDA max (months 13+) | 1.75 |
| `cash_basis_split` | Separate GAAP (covenants) from cash (waterfall) | 1 |

See `data/assumptions.csv` for the full parameter set (~80 configurable inputs).

## Sample Output

From a 10,000-path simulation over 60 months:

- **Covenant breach probability:** 0% across all covenants at all checkpoints (DSCR, D/EBITDA, current ratio, compensating balance)
- **Cash balance at month 24 (P50):** ~$3.4M | P5: ~$2.8M
- **TTM EBITDA at month 24 (P50):** ~$252K/month | P5: ~$168K/month
- **EBITDA negative probability:** ~18.9% (driven by early-period volatility in first 12 months, resolves by month 18)

## Technical Highlights

- **No hardcoded values.** Every financial assumption flows from the CSV configuration layer, making it fully portable across deal structures.
- **Dual-basis accounting.** GAAP figures drive covenant calculations while cash-basis figures drive the cash waterfall — matching how covenants are actually tested vs. how cash actually moves.
- **Cure analysis.** For any non-zero breach probability, the engine calculates the minimum revenue increase or expense decrease required to eliminate breaches at P0 (worst path), P5, P10, and P15 confidence levels.
- **Regression forecasting.** Revenue and expense projections use exponentially weighted seasonal decomposition with configurable half-life, rather than naive trailing averages.

## License

All Rights Reserved. See License for details. This code is shared for portfolio evaluation purposes only and may not be copied, modified, or used without explicit written consent.

## Author

Landon Webster — Controller / Finance & Analytics

