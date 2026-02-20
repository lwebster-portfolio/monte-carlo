
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Suppress expected NumPy warnings from percentile calculations on arrays with NaN/inf
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

# ============================================================
# Assumptions
# ============================================================
def read_assumptions(path: str = "assumptions.csv") -> Dict[str, str]:
    df = pd.read_csv(path).dropna(subset=["name"])
    df["name"] = df["name"].astype(str).str.strip()
    df = df[~df["name"].str.startswith("#")]
    # coerce values to string for downstream helpers
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        k = str(row["name"]).strip()
        if not k or k.lower() == "nan":
            continue
        out[k] = str(row.get("value", "")).split("#",1)[0].strip()
    return out


def load_revexp_series(revexp_path: str, horizon: int, start_month: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load GAAP revenue/expense baseline series from RevAndExp.csv.

    Expected columns: Month (e.g., 'Jan-20'), Revenue, Expenses, one_time_adjustment
    Returns: revenue_base_m, expenses_base_m, gaap_adjustment_m (all length = horizon, extended if needed)
    """
    df = pd.read_csv(revexp_path)
    req_cols = {"Month", "Revenue", "Expenses"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Rev/Exp CSV missing required columns: {sorted(missing)}")

    # Parse Month like 'Jan-20'
    df = df.copy()
    df["Month_dt"] = pd.to_datetime(df["Month"], format="%b-%y", errors="coerce")
    if df["Month_dt"].isna().any():
        bad = df.loc[df["Month_dt"].isna(), "Month"].head(5).tolist()
        raise ValueError(f"Rev/Exp CSV has unparseable Month values (expected 'Jan-20' style). Examples: {bad}")

    df = df.sort_values("Month_dt").reset_index(drop=True)

    if start_month:
        start_dt = pd.to_datetime(start_month, format="%b-%y")
        df = df.loc[df["Month_dt"] >= start_dt].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"start_month={start_month} is after the last month in Rev/Exp CSV")

    def _to_numeric_series(s: pd.Series) -> np.ndarray:
        # Handles commas, currency symbols, whitespace, and negatives in parentheses.
        ss = s.astype(str).str.strip()
        ss = ss.str.replace(",", "", regex=False).str.replace("$", "", regex=False)
        ss = ss.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        out = pd.to_numeric(ss, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return out

    rev = _to_numeric_series(df["Revenue"])
    exp = _to_numeric_series(df["Expenses"])
    if "one_time_adjustment" in df.columns:
        adj = _to_numeric_series(df["one_time_adjustment"])
    else:
        adj = np.zeros(len(df), dtype=float)

    # If CSV already has enough data (e.g., pre-extended with forecasts), use it directly
    if horizon <= len(rev):
        return rev[:horizon], exp[:horizon], adj[:horizon]

    # Otherwise need to extend - use forecast_mode to determine method
    rev_out = np.zeros(horizon, dtype=float)
    exp_out = np.zeros(horizon, dtype=float)
    adj_out = np.zeros(horizon, dtype=float)

    rev_out[:len(rev)] = rev
    exp_out[:len(exp)] = exp
    adj_out[:len(adj)] = adj

    tail_start = len(rev)
    pattern_len = min(12, len(rev))
    if pattern_len <= 0:
        raise ValueError("Rev/Exp CSV appears empty after filtering")

    # Default: repeat last 12-month pattern (static mode)
    for t in range(tail_start, horizon):
        k = (t - tail_start) % pattern_len
        rev_out[t] = rev[-pattern_len + k]
        exp_out[t] = exp[-pattern_len + k]
        adj_out[t] = 0.0

    return rev_out, exp_out, adj_out


def _compute_recency_weights(n_months: int, half_life_months: int = 12) -> np.ndarray:
    """Compute exponential recency weights for forecasting."""
    decay = 0.5 ** (1 / half_life_months)
    weights = decay ** np.arange(n_months)[::-1]
    return weights / weights.sum()


def _weighted_seasonal_factors(values: np.ndarray, cal_months: np.ndarray, half_life_months: int = 12) -> np.ndarray:
    """Compute weighted seasonal factors (12 values, one per calendar month)."""
    n = len(values)
    weights = _compute_recency_weights(n, half_life_months)
    weighted_mean = np.sum(values * weights)
    
    seasonal = np.ones(12)
    for m in range(1, 13):
        mask = cal_months == m
        if mask.sum() > 0:
            month_vals = values[mask]
            month_weights = weights[mask]
            month_weights = month_weights / month_weights.sum()
            seasonal[m-1] = np.sum(month_vals * month_weights) / weighted_mean if weighted_mean > 0 else 1.0
    
    return seasonal


def load_revexp_series_with_forecast(revexp_path: str, 
                                      horizon: int, 
                                      start_month: Optional[str] = None,
                                      expense_floor: float = 425000,
                                      expense_floor_hold_months: int = 9,
                                      expense_growth_annual: float = 0.03,
                                      half_life_months: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load GAAP revenue/expense with regression-based forecasting.
    
    Uses weighted seasonal decomposition + trend extrapolation:
    - Revenue: Linear trend with recency-weighted seasonality
    - Expenses: 
      1. Decay toward floor (capturing recent cost cuts)
      2. Hold at floor for expense_floor_hold_months (default 9 = through Q3 2026)
      3. Then grow at expense_growth_annual (default 3%) thereafter
    
    Uses FULL historical data for regression, then filters output to start_month.
    """
    df = pd.read_csv(revexp_path)
    req_cols = {"Month", "Revenue", "Expenses"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"Rev/Exp CSV missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Month_dt"] = pd.to_datetime(df["Month"], format="%b-%y", errors="coerce")
    df = df.sort_values("Month_dt").reset_index(drop=True)
    
    def _to_numeric_series(s: pd.Series) -> np.ndarray:
        ss = s.astype(str).str.strip()
        ss = ss.str.replace(",", "", regex=False).str.replace("$", "", regex=False)
        ss = ss.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        return pd.to_numeric(ss, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Use FULL history for regression
    rev_full = _to_numeric_series(df["Revenue"])
    exp_full = _to_numeric_series(df["Expenses"])
    
    if "one_time_adjustment" in df.columns:
        adj_full = _to_numeric_series(df["one_time_adjustment"])
    else:
        adj_full = np.zeros(len(df), dtype=float)

    n_hist = len(rev_full)
    month_idx = np.arange(n_hist)
    cal_months = df["Month_dt"].dt.month.values
    
    # Compute seasonal factors from FULL history
    rev_seasonal = _weighted_seasonal_factors(rev_full, cal_months, half_life_months)
    exp_seasonal = _weighted_seasonal_factors(exp_full, cal_months, half_life_months)
    
    # Deseasonalize
    rev_deseasonalized = rev_full / np.where(rev_seasonal[cal_months - 1] > 0, rev_seasonal[cal_months - 1], 1)
    exp_deseasonalized = exp_full / np.where(exp_seasonal[cal_months - 1] > 0, exp_seasonal[cal_months - 1], 1)
    
    # Revenue: weighted linear trend
    weights = _compute_recency_weights(n_hist, half_life_months)
    x_mean = np.sum(month_idx * weights)
    y_mean = np.sum(rev_deseasonalized * weights)
    xy_cov = np.sum(weights * (month_idx - x_mean) * (rev_deseasonalized - y_mean))
    x_var = np.sum(weights * (month_idx - x_mean) ** 2)
    rev_slope = xy_cov / x_var if x_var > 0 else 0
    rev_intercept = y_mean - rev_slope * x_mean
    
    # Expenses: estimate current level from recent data
    lookback = min(12, n_hist)
    recent_exp = exp_deseasonalized[-lookback:]
    recent_idx = month_idx[-lookback:]
    
    # Estimate current level (weighted toward most recent)
    recent_weights = _compute_recency_weights(lookback, half_life_months=6)
    exp_current = np.sum(recent_exp * recent_weights)
    
    # Estimate decay from slope of log(level - floor)
    above_floor = np.maximum(recent_exp - expense_floor, 1)
    log_above = np.log(above_floor)
    rx_mean = recent_idx.mean()
    ry_mean = log_above.mean()
    denom = np.sum((recent_idx - rx_mean)**2)
    slope = np.sum((recent_idx - rx_mean) * (log_above - ry_mean)) / denom if denom > 1e-9 else 0
    exp_decay = np.clip(np.exp(slope), 0.90, 1.02)
    
    # Determine start index
    if start_month:
        start_dt = pd.to_datetime(start_month, format="%b-%y")
        start_idx = df[df["Month_dt"] >= start_dt].index[0] if len(df[df["Month_dt"] >= start_dt]) > 0 else n_hist
    else:
        start_idx = 0
    
    last_hist_date = df["Month_dt"].iloc[-1]
    
    # Monthly expense growth factor (after hold period)
    exp_growth_monthly = (1 + expense_growth_annual) ** (1/12)
    
    # Build output arrays
    rev_out = np.zeros(horizon, dtype=float)
    exp_out = np.zeros(horizon, dtype=float)
    adj_out = np.zeros(horizon, dtype=float)
    
    for t in range(horizon):
        global_idx = start_idx + t  # Index in the full timeline
        
        if global_idx < n_hist:
            # Use historical data
            rev_out[t] = rev_full[global_idx]
            exp_out[t] = exp_full[global_idx]
            adj_out[t] = adj_full[global_idx]
        else:
            # Forecast
            months_from_hist = global_idx - n_hist + 1
            forecast_date = last_hist_date + pd.DateOffset(months=months_from_hist)
            cal_month = forecast_date.month
            
            # Revenue forecast (linear trend)
            rev_trend = rev_intercept + rev_slope * global_idx
            rev_out[t] = max(rev_trend * rev_seasonal[cal_month - 1], 0)
            
            # Expense forecast: decay -> hold at floor -> then grow
            if months_from_hist <= expense_floor_hold_months:
                # Phase 1 & 2: Decay toward floor and hold
                exp_trend = expense_floor + max(0, (exp_current - expense_floor) * (exp_decay ** months_from_hist))
                exp_trend = max(exp_trend, expense_floor)  # Don't go below floor
            else:
                # Phase 3: Grow from floor
                months_growing = months_from_hist - expense_floor_hold_months
                exp_trend = expense_floor * (exp_growth_monthly ** months_growing)
            
            # Apply seasonality, enforce floor
            exp_out[t] = max(exp_trend * exp_seasonal[cal_month - 1], expense_floor * 0.95)
            adj_out[t] = 0.0
    
    return rev_out, exp_out, adj_out

def load_new_client_revenue(csv_path: str,
                            horizon: int,
                            gaap_start_month: Optional[str] = None,
                            realization_pct: float = 1.0) -> np.ndarray:
    """
    Load new client net revenue overlay from CSV.

    CSV format: Month (Mon-YY), total_net (and optionally per-client columns).
    Returns array of length `horizon`, aligned to the GAAP start month.
    Months before the CSV start are 0. Months after the CSV end are held flat at the last value.

    realization_pct: Scale factor (1.0 = full pipeline, 0.5 = half materializes).
    """
    df = pd.read_csv(csv_path)
    if "Month" not in df.columns or "total_net" not in df.columns:
        raise ValueError(f"New client CSV must have 'Month' and 'total_net' columns. Got: {list(df.columns)}")

    df = df.copy()
    df["Month_dt"] = pd.to_datetime(df["Month"], format="%b-%y", errors="coerce")
    if df["Month_dt"].isna().any():
        bad = df.loc[df["Month_dt"].isna(), "Month"].head(5).tolist()
        raise ValueError(f"New client CSV has unparseable Month values. Examples: {bad}")

    df = df.sort_values("Month_dt").reset_index(drop=True)
    net = pd.to_numeric(df["total_net"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Determine GAAP timeline start
    if gaap_start_month:
        gaap_start_dt = pd.to_datetime(gaap_start_month, format="%b-%y")
    else:
        gaap_start_dt = df["Month_dt"].iloc[0]

    # Map CSV months to GAAP-timeline indices
    csv_start_dt = df["Month_dt"].iloc[0]
    # Months offset from GAAP start to CSV start
    offset = (csv_start_dt.year - gaap_start_dt.year) * 12 + (csv_start_dt.month - gaap_start_dt.month)

    out = np.zeros(horizon, dtype=float)
    for j in range(len(net)):
        idx = offset + j
        if 0 <= idx < horizon:
            out[idx] = net[j]

    # Fill months after CSV end with last value (flat-line)
    last_csv_idx = offset + len(net) - 1
    if last_csv_idx < horizon - 1 and len(net) > 0:
        fill_val = net[-1]
        out[last_csv_idx + 1:] = fill_val

    return out * float(realization_pct)


def f(A: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(A.get(key, default))
    except Exception:
        return float(default)

def i(A: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(A.get(key, default)))
    except Exception:
        return int(default)

def s(A: Dict[str, str], key: str, default: str = "") -> str:
    v = A.get(key, default)
    return str(v).strip()

# ============================================================
# Formatting
# ============================================================
def fmt_money(x) -> str:
    try:
        if not np.isfinite(x):
            return "NA"
        return f"${float(x):,.0f}"
    except Exception:
        return "NA"

def fmt_num(x, digits=2) -> str:
    try:
        if not np.isfinite(x):
            return "NA"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "NA"

def fmt_pct(x) -> str:
    try:
        if not np.isfinite(x):
            return "NA"
        return f"{100*float(x):.1f}%"
    except Exception:
        return "NA"

# ============================================================
# Safe stats
# ============================================================
def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]

def safe_percentile(a, p, default=np.nan) -> float:
    x = _finite(a)
    if x.size == 0:
        return float(default)
    return float(np.percentile(x, p))

def safe_median(a, default=np.nan) -> float:
    x = _finite(a)
    if x.size == 0:
        return float(default)
    return float(np.median(x))

def safe_prob(a, predicate, default=np.nan) -> float:
    x = _finite(a)
    if x.size == 0:
        return float(default)
    return float(np.mean(predicate(x)))

# ============================================================
# Stochastic helpers
# ============================================================
def jitter_curve(weights: np.ndarray, concentration: float = 200.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Dirichlet jitter around a base curve.
    weights must be a probability vector (sums to 1).

    rng: optional numpy Generator to avoid horizon-dependent RNG consumption.
    """
    if rng is None:
        rng = np.random.default_rng()
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()
    alpha = w * max(float(concentration), 1e-6)
    return rng.dirichlet(alpha)

def draw_lognormal_series(mean_arr: np.ndarray, cv: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Draw a lognormal series with elementwise means and constant coefficient of variation (cv)."""
    if rng is None:
        rng = np.random.default_rng()
    mean_arr = np.asarray(mean_arr, dtype=float)
    cv = float(cv)
    if cv <= 1e-12:
        return mean_arr.copy()
    sigma2 = np.log1p(cv * cv)
    sigma = np.sqrt(sigma2)
    mu = np.log(np.clip(mean_arr, 1e-12, None)) - 0.5 * sigma2
    return rng.lognormal(mean=mu, sigma=sigma, size=mean_arr.shape)

def draw_multiplicative_lognormal(base_arr: np.ndarray, cv: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply multiplicative lognormal noise (mean 1) to a base series, preserving sign and zeros."""
    if rng is None:
        rng = np.random.default_rng()
    base_arr = np.asarray(base_arr, dtype=float)
    cv = float(cv)
    if cv <= 1e-12:
        return base_arr.copy()
    sigma2 = np.log1p(cv * cv)
    sigma = np.sqrt(sigma2)
    mu = -0.5 * sigma2  # mean of factor = 1
    factor = rng.lognormal(mean=mu, sigma=sigma, size=base_arr.shape)
    return base_arr * factor

def lognormal_from_mean_cv(mean: float, cv: float, rng: Optional[np.random.Generator] = None) -> float:
    """Draw a single lognormal value given target mean and coefficient of variation."""
    if rng is None:
        rng = np.random.default_rng()
    mean = float(mean)
    cv = float(cv)
    if cv <= 1e-12:
        return float(mean)
    sigma2 = np.log1p(cv * cv)
    sigma = float(np.sqrt(sigma2))
    mu = float(np.log(max(mean, 1e-12)) - 0.5 * sigma2)
    return float(rng.lognormal(mean=mu, sigma=sigma))

# ============================================================
# Finance helpers
# ============================================================
def amort_payment(principal: float, apr: float, n_months: int) -> float:
    r = apr / 12.0
    if n_months <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return principal / n_months
    return principal * (r * (1 + r) ** n_months) / ((1 + r) ** n_months - 1)

def trailing_sum(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    if window <= 1:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[window - 1:] = c[window:] - c[:-window]
    return out

def scheduled_principal_next12(balance: float,
                               apr: float,
                               amort_months: int,
                               io_months: int,
                               m: int) -> float:
    """
    'Current maturities of LTD' proxy: sum of scheduled principal in months (m+1 .. m+12),
    based on *current balance* and remaining term structure.

    Assumptions:
      - IO months occur first. During IO, payment = interest only, principal = 0.
      - After IO, payment is level amortizing over amort_months (the amort period AFTER IO).
      - We recompute payment once at the moment amort begins from the then-balance and
        remaining amort months.
    """
    bal = float(max(balance, 0.0))
    if bal <= 1e-9:
        return 0.0

    # months completed through m (0-indexed)
    months_done = m + 1
    remaining_io = max(0, int(io_months) - months_done)
    amort_elapsed = max(0, months_done - int(io_months))
    remaining_amort = max(0, int(amort_months) - amort_elapsed)

    r = float(apr) / 12.0
    prin_sum = 0.0

    # simulate forward 12 months
    for k in range(12):
        if bal <= 1e-9:
            break

        if remaining_io > 0:
            # IO month: principal 0
            remaining_io -= 1
            # balance unchanged
            continue

        if remaining_amort <= 0:
            # No scheduled amort remaining (balloon not modeled)
            break

        pmt = amort_payment(bal, apr, remaining_amort)
        interest = bal * r
        principal = max(0.0, min(bal, pmt - interest))
        prin_sum += principal
        bal -= principal
        remaining_amort -= 1

    return float(prin_sum)

# ============================================================
# Covenant calculations (point-in-time values)
# ============================================================
def borrower_dscr_value(ebitda_m: np.ndarray,
                        borrower_interest_m: np.ndarray,
                        curr_maturities: float,
                        m: int,
                        A: Dict[str, str]) -> float:
    """
    Borrower DSCR at month m (quarter-end), per term sheet:

      (EBITDA Ã¢â‚¬â€œ Cash Taxes Ã¢â‚¬â€œ Dividends Ã¢â‚¬â€œ Distributions)
      /
      (Current Maturities of Long Term Debt + Interest Expense)

    Window rule:
      - YTD until 12 months exist
      - then trailing 12 months thereafter
    """
    taxes_m = f(A, "cash_taxes_annual", 0.0) / 12.0
    div_m = f(A, "dividends_annual", 0.0) / 12.0
    dist_m = f(A, "distributions_annual", 0.0) / 12.0

    adj_ebitda = ebitda_m - (taxes_m + div_m + dist_m)

    window = min(m + 1, 12)
    start = m - window + 1

    num = float(np.sum(adj_ebitda[start:m+1]))
    int_sum = float(np.sum(borrower_interest_m[start:m+1]))

    # If we only have partial-year history, treat DSCR as "annualized YTD" against a 12-month debt-service burden.
    # This avoids mechanically failing Q1/Q2/Q3 simply because the numerator window is shorter than the forward-looking
    # current maturities concept.
    if window < 12:
        scale = 12.0 / float(window)
        num *= scale
        int_sum *= scale

    # External debt service (existing borrower debt not refinanced by lender)
    ext_ds_annual = f(A, "ext_debt_annual_ds", 0.0)
    
    den = float(int_sum) + float(curr_maturities) + float(ext_ds_annual)
    if not np.isfinite(num) or den <= 1e-9:
        return np.nan
    return float(num / den)

def debt_to_ebitda_value(ebitda_m: np.ndarray, funded_debt_bal_m: np.ndarray, m: int) -> float:
    """Funded Debt / EBITDA at month m.
    Uses annualized YTD until 12 months exist, then trailing-12 thereafter.
    """
    window = min(m + 1, 12)
    start = m - window + 1
    e_sum = float(np.sum(ebitda_m[start:m+1]))
    if window < 12:
        e_sum *= (12.0 / float(window))
    if not np.isfinite(e_sum) or e_sum <= 1e-9:
        return np.nan
    return float(funded_debt_bal_m[m] / e_sum)

def global_dscr_value(ebitda_m: np.ndarray,
                      borrower_interest_m: np.ndarray,
                      curr_maturities: float,
                      personal_ds_m: np.ndarray,
                      m: int,
                      debt_mode: str,
                      A: Dict[str, str]) -> float:
    """
    Global DSCR at annual test months (12,24,...), per term sheet:

      (EBITDA Ã¢â‚¬â€œ Cash Taxes Ã¢â‚¬â€œ Dividends Ã¢â‚¬â€œ Distributions) + (Personal Cash Flow Available to Service Debt)
      /
      (Current Maturities of Long Term Debt + Interest Expense) + (Required Principal and Interest Payments on Personal Debts)

    Interpretation rules:
      - 4M: include personal debt service (simulated) and personal cash flow add-back.
      - 2M: exclude personal debt and personal cash flow (remove $2.5M personal loan from global covenant).

    Optional assumptions:
      - personal_cash_flow_annual (default 0)
      - extra_personal_ds_annual (default 0)
    """
    if m < 11:
        return np.nan  # annual tests require 12 months of history

    taxes_m = f(A, "cash_taxes_annual", 0.0) / 12.0
    div_m = f(A, "dividends_annual", 0.0) / 12.0
    dist_m = f(A, "distributions_annual", 0.0) / 12.0

    adj_ebitda = ebitda_m - (taxes_m + div_m + dist_m)

    start = m - 11
    num = float(np.sum(adj_ebitda[start:m+1]))
    int_sum = float(np.sum(borrower_interest_m[start:m+1]))
    den = float(int_sum) + float(curr_maturities)

    if str(debt_mode).upper() == "4M":
        num += float(f(A, "personal_cash_flow_annual", 0.0))
        den += float(np.sum(personal_ds_m[start:m+1])) + float(f(A, "extra_personal_ds_annual", 0.0))

    if not np.isfinite(num) or den <= 1e-9:
        return np.nan
    return float(num / den)

def current_ratio_value(cash: float,
                        restricted_cash_balance: float,
                        include_restricted_cash_in_ca: bool,
                        portfolio_collections_m: np.ndarray,
                        m: int,
                        other_ca: float,
                        other_cl: float,
                        portfolio_current_months: int,
                        portfolio_current_haircut: float,
                        include_rloc_in_cl: bool,
                        rloc_balance: float,
                        curr_maturities: float) -> float:
    """
    Bank-style current ratio:
      CA = cash + other_current_assets + restricted_cash (if included) + next N months portfolio collections (haircut)
      CL = other_current_liabilities + current maturities (next 12 principal) + RLOC balance (if included)
    """
    ca = float(cash) + float(other_ca)
    if include_restricted_cash_in_ca:
        ca += float(restricted_cash_balance)

    start = m + 1
    end = min(len(portfolio_collections_m), start + max(0, int(portfolio_current_months)))
    if start < end:
        ca += float(portfolio_current_haircut) * float(np.sum(portfolio_collections_m[start:end]))

    cl = float(other_cl) + float(curr_maturities)
    if include_rloc_in_cl:
        cl += float(rloc_balance)

    if cl <= 1e-9:
        return float("inf")
    return float(ca / cl)

# ============================================================
# Simulation
# ============================================================
def run_sim(curve_path: str = "curve.csv",
            assumptions_path: str = "assumptions.csv",
            revexp_path: str = "RevAndExp.csv",
            seed: Optional[int] = None,
            checkpoint_months: Optional[List[int]] = None,
            verbose_close_check: bool = True,
            ):
    if seed is not None:
        np.random.seed(int(seed))

    A = read_assumptions(assumptions_path)

    debt_mode = s(A, "debt_mode", "2M").upper()  # 2M or 4M
    if debt_mode not in {"2M", "4M"}:
        raise ValueError("assumptions: debt_mode must be '2M' or '4M'.")

    horizon = i(A, "horizon_months", 36)
    n_sims = i(A, "n_sims", 10000)

    checkpoint_months = checkpoint_months or [12, 18, 24, 36]
    checkpoint_months = [m for m in checkpoint_months if 1 <= m <= horizon]

    # ----------------- Curve (FIXED economic horizon) -----------------
    curve = pd.read_csv(curve_path)
    if "weight" not in curve.columns:
        raise ValueError("curve.csv must contain a 'weight' column.")

    base_w_all = curve["weight"].astype(float).to_numpy()
    base_w_all = np.clip(base_w_all, 0.0, None)

    curve_h = i(A, "portfolio_curve_horizon_months", min(36, len(base_w_all)))
    # Cap curve horizon to both available curve weights AND simulation horizon so extending the curve
    # never breaks (or changes) earlier-month cashflows.
    curve_h = max(1, int(curve_h))
    curve_h = min(curve_h, len(base_w_all), horizon)

    base_w = base_w_all[:curve_h]
    if base_w.sum() <= 0:
        raise ValueError("curve weights sum to 0 for portfolio_curve_horizon_months.")
    base_w = base_w / base_w.sum()

    curve_conc = f(A, "curve_jitter_concentration", 200.0)
    # Tail mode removed (explicit 60-month curve term; no behavior beyond curve_h).

    # ----------------- Operating base -----------------
    accounting_basis = str(A.get("accounting_basis", "cash")).strip().lower().strip(",")
    gaap_start_month = A.get("gaap_start_month", "").strip() or None
    gaap_adjustment_is_cash = (i(A, "gaap_adjustment_is_cash", 0) == 1)

    rev_cv = f(A, "revenue_sigma", 0.0)
    exp_cv = f(A, "expenses_sigma", 0.0)
    growth_m = (1.0 + f(A, "business_growth_annual", 0.0)) ** (1/12) - 1.0
    expense_growth_annual = f(A, "expense_growth_annual", 0.03)

    month_idx = np.arange(horizon, dtype=float)

    gaap_adj_m = np.zeros(horizon, dtype=float)

    if accounting_basis == "gaap":
        # Load baseline GAAP series from RevAndExp.csv
        # If revexp_path is relative, resolve relative to assumptions.csv directory.
        revexp_resolved = revexp_path
        if not os.path.isabs(revexp_resolved):
            revexp_resolved = os.path.join(os.path.dirname(os.path.abspath(assumptions_path)), revexp_resolved)

        # Forecast mode: 'static' (repeat last 12mo pattern) or 'regression' (weighted seasonal + trend)
        forecast_mode = s(A, "forecast_mode", "static").lower()
        expense_floor = f(A, "expense_floor", 425000)
        expense_floor_hold_months = i(A, "expense_floor_hold_months", 9)
        forecast_half_life = i(A, "forecast_half_life_months", 12)

        if forecast_mode == "regression":
            gaap_rev_base, gaap_exp_base, gaap_adj = load_revexp_series_with_forecast(
                revexp_resolved, 
                horizon=horizon, 
                start_month=gaap_start_month,
                expense_floor=expense_floor,
                expense_floor_hold_months=expense_floor_hold_months,
                expense_growth_annual=expense_growth_annual,
                half_life_months=forecast_half_life
            )
            # Regression mode already builds in trend - don't double-apply growth/trend
            rev_mean = gaap_rev_base
            exp_mean = gaap_exp_base
        else:
            # Static mode: use flat extension + manual growth/trend overlays
            gaap_rev_base, gaap_exp_base, gaap_adj = load_revexp_series(
                revexp_resolved, horizon=horizon, start_month=gaap_start_month
            )
            # Apply revenue growth to GAAP baseline (always on - set to 0 to disable)
            growth_factor = ((1 + growth_m) ** month_idx)
            rev_mean = gaap_rev_base * growth_factor

            # Apply expense growth (uses same parameter as regression mode)
            exp_growth_m = (1.0 + expense_growth_annual) ** (1/12) - 1.0
            exp_mean = gaap_exp_base * ((1 + exp_growth_m) ** month_idx)

        gaap_adj_m = gaap_adj

    else:
            
        # Cash basis: static starting point + growth
        base_rev = f(A, "base_cash_revenue_monthly", 0.0)
        base_exp = f(A, "base_cash_expenses_monthly", 0.0)

        rev_mean = base_rev * ((1 + growth_m) ** month_idx)
        exp_mean = base_exp * ((1 + growth_m) ** month_idx)
        if abs(expense_growth_annual) > 1e-12:
            exp_growth_m = (1.0 + expense_growth_annual) ** (1/12) - 1.0
            exp_mean = exp_mean * ((1 + exp_growth_m) ** month_idx)

    # ----------------- New Client Revenue Overlay -----------------
    new_client_enabled = (i(A, "new_client_enabled", 0) == 1)
    new_client_base_m = np.zeros(horizon, dtype=float)
    new_client_cv = 0.0

    if new_client_enabled:
        nc_csv = s(A, "new_client_csv", "new_client_revenue.csv")
        if not os.path.isabs(nc_csv):
            nc_csv = os.path.join(os.path.dirname(os.path.abspath(assumptions_path)), nc_csv)
        nc_realization = f(A, "new_client_realization_pct", 1.0)
        new_client_cv = f(A, "new_client_sigma", 0.0)

        new_client_base_m = load_new_client_revenue(
            csv_path=nc_csv,
            horizon=horizon,
            gaap_start_month=gaap_start_month,
            realization_pct=nc_realization,
        )

        if verbose_close_check:
            nc_total = float(np.sum(new_client_base_m))
            nc_peak = float(np.max(new_client_base_m))
            nc_first_nonzero = int(np.argmax(new_client_base_m > 0)) + 1 if np.any(new_client_base_m > 0) else 0
            print(f"New client overlay:    ENABLED (realization={nc_realization:.0%}, CV={new_client_cv:.2f})")
            print(f"  Total over horizon:  {fmt_money(nc_total)}")
            print(f"  Peak monthly:        {fmt_money(nc_peak)}")
            print(f"  First active month:  {nc_first_nonzero}")

    # ----------------- Cash Basis Split -----------------
    # When enabled, covenant EBITDA uses GAAP basis (rev_mean, exp_mean) while
    # the cash waterfall uses a separate cash-basis baseline (actual cash collections
    # and cash disbursements from the CF forecast). This prevents GAAP accrual timing
    # from inflating or deflating the ending book cash position.
    #
    # cash_inflow[m] = cash_rev[m] - cash_exp[m] + portfolio_collections[m] + new_client[m] + step_gains
    # ebitda_covenant[m] = gaap_rev[m] - gaap_exp[m] + port_for_pnl[m] + new_client[m] + step_gains + gaap_adj
    #
    cash_basis_split = (i(A, "cash_basis_split", 0) == 1)
    cash_rev_base = np.zeros(horizon, dtype=float)
    cash_exp_base = np.zeros(horizon, dtype=float)
    cash_basis_rev_cv = f(A, "cash_basis_rev_sigma", 0.0)  # defaults to 0 (use revenue_sigma if not set)
    cash_basis_exp_cv = f(A, "cash_basis_exp_sigma", 0.0)

    if cash_basis_split:
        cb_csv = s(A, "cash_basis_csv", "cash_basis_baseline.csv")
        if not os.path.isabs(cb_csv):
            cb_csv = os.path.join(os.path.dirname(os.path.abspath(assumptions_path)), cb_csv)

        cb_df = pd.read_csv(cb_csv)
        req = {"month", "cash_revenue", "cash_expenses"}
        missing = req - set(cb_df.columns)
        if missing:
            raise ValueError(f"cash_basis_baseline.csv missing columns: {sorted(missing)}")

        cb_df = cb_df.sort_values("month").reset_index(drop=True)
        n_cb = len(cb_df)

        cb_rev_arr = cb_df["cash_revenue"].astype(float).to_numpy()
        cb_exp_arr = cb_df["cash_expenses"].astype(float).to_numpy()

        # Fill horizon: use CSV data directly, then repeat last 12-month pattern if needed
        if n_cb >= horizon:
            cash_rev_base[:horizon] = cb_rev_arr[:horizon]
            cash_exp_base[:horizon] = cb_exp_arr[:horizon]
        else:
            cash_rev_base[:n_cb] = cb_rev_arr
            cash_exp_base[:n_cb] = cb_exp_arr
            pattern_len = min(12, n_cb)
            for t in range(n_cb, horizon):
                k = (t - n_cb) % pattern_len
                cash_rev_base[t] = cb_rev_arr[-pattern_len + k]
                cash_exp_base[t] = cb_exp_arr[-pattern_len + k]

        # Default stochastic CVs to main revenue/expense sigmas if not explicitly set
        if cash_basis_rev_cv <= 1e-12:
            cash_basis_rev_cv = rev_cv
        if cash_basis_exp_cv <= 1e-12:
            cash_basis_exp_cv = exp_cv

        if verbose_close_check:
            print(f"Cash basis split:      ENABLED")
            print(f"  Cash rev avg M1-12:  {fmt_money(cash_rev_base[:12].mean())}/mo")
            print(f"  Cash exp avg M1-12:  {fmt_money(cash_exp_base[:12].mean())}/mo")
            print(f"  Cash net avg M1-12:  {fmt_money((cash_rev_base[:12] - cash_exp_base[:12]).mean())}/mo")
            print(f"  Stochastic CV (rev): {cash_basis_rev_cv:.2f}")
            print(f"  Stochastic CV (exp): {cash_basis_exp_cv:.2f}")

    # ----------------- Portfolio -----------------
    base_total_col = f(A, "base_total_collections", 0.0)
    col_cv = f(A, "collections_sigma", 0.0)

    monthly_step_gain = (
        f(A, "gain_paydown_direct_monthly", 0.0)
        + f(A, "gain_payoff_ap_monthly", 0.0)
        + f(A, "gain_absorb_loan_monthly", 0.0)
        + f(A, "gain_absorb_loc_monthly", 0.0)
    )

    # ----------------- Debt terms -----------------
    term_io_months = i(A, "term_io_months", 0)

    loan1_principal0 = f(A, "loan1_principal", 0.0)
    loan1_apr = f(A, "loan1_apr", 0.0)
    loan1_amort = i(A, "loan1_amort_months", 60)

    loan2_principal0 = f(A, "loan2_principal", 0.0)
    loan2_apr = f(A, "loan2_apr", 0.0)
    loan2_amort = i(A, "loan2_amort_months", 84)

    # Amortization interpretation:
    # - If loan*_amort_months_after_io is provided, use it.
    # - Else treat loan*_amort_months as TOTAL term length (including IO), so amort-after-IO = max(1, total - IO).
    loan1_amort_after_io = i(A, "loan1_amort_months_after_io", 0)
    if loan1_amort_after_io <= 0:
        loan1_amort_after_io = max(1, int(loan1_amort) - int(term_io_months))
    loan2_amort_after_io = i(A, "loan2_amort_months_after_io", 0)
    if loan2_amort_after_io <= 0:
        loan2_amort_after_io = max(1, int(loan2_amort) - int(term_io_months))

    # Fixed amortizing payments once amortization begins
    loan1_pmt_fixed = amort_payment(loan1_principal0, loan1_apr, loan1_amort_after_io) if loan1_amort_after_io > 0 else 0.0
    loan2_pmt_fixed = amort_payment(loan2_principal0, loan2_apr, loan2_amort_after_io) if loan2_amort_after_io > 0 else 0.0

    # Revolver
    loc_limit = f(A, "loc_limit", 0.0)
    loc_apr = f(A, "loc_apr", 0.0)
    loc_r = loc_apr / 12.0
    loc_initial_draw = f(A, "loc_initial_draw", 0.0)
    loc_auto_paydown = (i(A, "loc_auto_paydown", 1) == 1)
    loc_draw_to_prevent_cash_negative = (i(A, "loc_draw_to_prevent_cash_negative", 0) == 1)
    loc_min_cash_buffer = f(A, "loc_min_cash_buffer", 0.0)

    # RLOC annual rest
    rest_enabled = (i(A, "rloc_rest_enabled", 0) == 1)
    rest_month_of_year = i(A, "rloc_rest_month_of_year", 12)
    rest_duration = i(A, "rloc_rest_duration_months", 1)

    rest_block_months = set()
    if rest_enabled:
        for y in range(1, horizon // 12 + 3):
            start_m = (y - 1) * 12 + rest_month_of_year
            for k in range(rest_duration):
                mm = start_m + k
                if 1 <= mm <= horizon:
                    rest_block_months.add(mm)

    # ----------------- Close check -----------------
    one_time_uses = (
        f(A, "purchase_price", 0.0)
        + f(A, "use_paydown_direct", 0.0)
        + f(A, "use_payoff_ap", 0.0)
        + f(A, "use_absorb_loan_payoff", 0.0)
        + f(A, "use_absorb_loc_payoff", 0.0)
    )
    term_proceeds = f(A, "total_new_term_debt_proceeds", 0.0)
    reserve_initial = f(A, "interest_reserve_initial", 0.0)

    starting_cash = (term_proceeds + loc_initial_draw) - one_time_uses - reserve_initial + f(A, "starting_cash_adjustment", 0.0)

    # Non-covenant cash outflows (external debt service, professional fees, taxes, etc.)
    # These reduce cash but do NOT affect EBITDA or covenant calculations.
    other_cash_disbursements_monthly = f(A, "other_cash_disbursements_monthly", 0.0)

    # ----------------- Covenants / toggles -----------------
    comp_pct = f(A, "comp_balance_pct", 0.20)
    comp_balance_cash_eligible = (i(A, "comp_balance_cash_eligible", 1) == 1)
    comp_include_personal = (i(A, "comp_balance_include_personal", 1 if debt_mode=="4M" else 0) == 1)
    avg_trust_balance = f(A, "avg_trust_balance", 0.0)
    trust_haircut = f(A, "trust_haircut", 1.0)

    current_ratio_min = f(A, "current_ratio_min", 1.25)
    other_ca = f(A, "other_current_assets", 0.0)
    other_cl = f(A, "other_current_liabilities", 0.0)
    portfolio_current_months = i(A, "portfolio_current_months", 12)
    portfolio_current_haircut = f(A, "portfolio_current_haircut", 1.0)

    include_rloc_in_cl = (i(A, "current_ratio_include_rloc_in_cl", 1) == 1)
    include_restricted_cash_in_ca = (i(A, "current_ratio_include_restricted_cash", 1) == 1)
    restricted_cash_balance = f(A, "restricted_cash_balance", 0.0)

    dscr_min = f(A, "dscr_min", 1.30)
    gdscr_min = f(A, "gdscr_min", 2.0)
    personal_ds_annual = f(A, "personal_debt_service_annual", 0.0)

    dte_max_12 = f(A, "debt_to_ebitda_max_first12", 2.25)
    dte_max_after = f(A, "debt_to_ebitda_max_after12", 1.75)

    if verbose_close_check:
        print("---- CLOSE CHECK ----")
        print(f"Debt mode:             {debt_mode}")
        print(f"Term proceeds:         {fmt_money(term_proceeds)}")
        print(f"LoC initial draw:      {fmt_money(loc_initial_draw)} (limit {fmt_money(loc_limit)}, APR {loc_apr*100:.2f}%)")
        print(f"Interest reserve:      {fmt_money(reserve_initial)}")
        print(f"One-time uses:         {fmt_money(one_time_uses)}")
        print(f"Starting cash:         {fmt_money(starting_cash)}")
        print(f"Undrawn LoC capacity:  {fmt_money(loc_limit - loc_initial_draw)}")
        if avg_trust_balance > 0:
            print(f"Average Monthly Trust Balance: {fmt_money(avg_trust_balance)} (haircut {trust_haircut:.2f})")
        if other_cash_disbursements_monthly > 0:
            print(f"Other cash disbursements: {fmt_money(other_cash_disbursements_monthly)}/mo ({fmt_money(other_cash_disbursements_monthly * 12)}/yr)")
        print("---------------------\n")
        if starting_cash < 0:
            print(f"WARNING: Starting cash is NEGATIVE ({fmt_money(starting_cash)}). Close does not fund as modeled.\n")

    # ----------------- Outputs -----------------
    cash_ts = np.full((n_sims, horizon), np.nan, dtype=float)
    ebitda_ts = np.full((n_sims, horizon), np.nan, dtype=float)

    # Store underlying stochastic draws so we can compute required rev/expense changes fast
    revenue_ts = np.full((n_sims, horizon), np.nan, dtype=float)
    opex_ts    = np.full((n_sims, horizon), np.nan, dtype=float)
    portcol_ts = np.full((n_sims, horizon), np.nan, dtype=float)

    # Store debt/interest drivers needed for analytic covenant solve
    borrower_interest_ts = np.full((n_sims, horizon), np.nan, dtype=float)
    personal_ds_ts = np.full((n_sims, horizon), np.nan, dtype=float)
    funded_debt_bal_ts   = np.full((n_sims, horizon), np.nan, dtype=float)
    curr_maturities_ts   = np.full((n_sims, horizon), np.nan, dtype=float)  # populated only at test months

    # Loan balance tracking (for monthly output)
    loan1_bal_ts = np.full((n_sims, horizon), np.nan, dtype=float)
    loan2_bal_ts = np.full((n_sims, horizon), np.nan, dtype=float)
    loc_bal_ts   = np.full((n_sims, horizon), np.nan, dtype=float)

    # Values at test months
    cr_val_ts = np.full((n_sims, horizon), np.nan, dtype=float)
    dscr_val_ts = np.full((n_sims, horizon), np.nan, dtype=float)
    dte_val_ts = np.full((n_sims, horizon), np.nan, dtype=float)
    gdscr_val_ts = np.full((n_sims, horizon), np.nan, dtype=float)

    # Breach flags at test months
    comp_breach_ts = np.full((n_sims, horizon), np.nan, dtype=float)  # monthly
    cr_breach_ts   = np.full((n_sims, horizon), np.nan, dtype=float)  # QE
    dscr_breach_ts = np.full((n_sims, horizon), np.nan, dtype=float)  # QE >=12
    dte_breach_ts  = np.full((n_sims, horizon), np.nan, dtype=float)  # QE >=12
    gdscr_breach_ts= np.full((n_sims, horizon), np.nan, dtype=float)  # AE
    rest_breach_ts = np.full((n_sims, horizon), np.nan, dtype=float)  # monthly (rest months)

    # Independent RNG streams per component to avoid horizon-length RNG consumption changing early-month results.
    master_ss = np.random.SeedSequence(int(seed) if seed is not None else None)

    for sim in range(n_sims):
        sim_ss = master_ss.spawn(1)[0]
        rng_port = np.random.default_rng(sim_ss.spawn(1)[0])
        rng_rev  = np.random.default_rng(sim_ss.spawn(1)[0])
        rng_exp  = np.random.default_rng(sim_ss.spawn(1)[0])
        rng_cash_rev = np.random.default_rng(sim_ss.spawn(1)[0])  # cash basis revenue
        rng_cash_exp = np.random.default_rng(sim_ss.spawn(1)[0])  # cash basis expenses

        # ---- Portfolio collections (curve anchored to curve_h, not horizon) ----
        total_col = lognormal_from_mean_cv(base_total_col, col_cv, rng=rng_port)

        w = jitter_curve(base_w, concentration=curve_conc, rng=rng_port)
        port_col = np.zeros(horizon, dtype=float)
        if curve_h > 0:
            n_w = min(curve_h, horizon)
            port_col[:n_w] = total_col * w[:n_w]

        # Revenue / operating expenses
        if accounting_basis == "gaap":
            revenue = draw_multiplicative_lognormal(rev_mean, rev_cv, rng=rng_rev)
            opex    = draw_multiplicative_lognormal(exp_mean, exp_cv, rng=rng_exp)
        else:
            revenue = draw_lognormal_series(rev_mean, rev_cv, rng=rng_rev)
            opex    = draw_lognormal_series(exp_mean, exp_cv, rng=rng_exp)

        # store draws for analytic reverse-solve
        revenue_ts[sim, :] = revenue
        opex_ts[sim, :]    = opex
        portcol_ts[sim, :] = port_col

        # EBITDA (monthly)
        gaap_rev_includes_col = (i(A, "gaap_revenue_includes_collections", 0) == 1)
        port_for_pnl = np.zeros_like(port_col) if (accounting_basis == "gaap" and gaap_rev_includes_col) else port_col

        # New client revenue overlay (deterministic base + optional stochastic noise)
        if new_client_enabled and np.any(new_client_base_m > 0):
            new_client_sim = draw_multiplicative_lognormal(new_client_base_m, new_client_cv, rng=rng_rev)
            # Zero out months where base is zero (no noise on zero)
            new_client_sim = np.where(new_client_base_m > 0, new_client_sim, 0.0)
        else:
            new_client_sim = np.zeros(horizon, dtype=float)

        ebitda_covenant = (revenue - opex + port_for_pnl) + monthly_step_gain + gaap_adj_m + new_client_sim
        if cash_basis_split:
            # Cash waterfall uses CF-derived cash collections and cash expenses
            # Portfolio collections (vendor curve) and new client are additive to both
            cash_rev_sim = draw_multiplicative_lognormal(cash_rev_base, cash_basis_rev_cv, rng=rng_cash_rev)
            cash_exp_sim = draw_multiplicative_lognormal(cash_exp_base, cash_basis_exp_cv, rng=rng_cash_exp)
            ebitda_cash = (cash_rev_sim - cash_exp_sim + port_col) + monthly_step_gain + new_client_sim
        else:
            ebitda_cash = (revenue - opex + port_for_pnl) + monthly_step_gain + (gaap_adj_m if gaap_adjustment_is_cash else 0.0) + new_client_sim
        ebitda = ebitda_covenant
        ebitda_ts[sim, :] = ebitda_covenant

        # Balances
        cash = float(starting_cash)
        loan1_bal = float(loan1_principal0)
        loan2_bal = float(loan2_principal0)
        loc_bal = float(loc_initial_draw)

        borrower_interest_m = np.zeros(horizon, dtype=float)  # for Borrower DSCR (mode-dependent)
        borrower_entity_interest_m = np.zeros(horizon, dtype=float)  # for Global DSCR (borrower-only interest)
        funded_debt_bal_m = np.zeros(horizon, dtype=float)
        personal_ds_m = np.zeros(horizon, dtype=float)  # personal debt service paid by company (for Global DSCR in 4M)

        for m in range(horizon):
            month_num = m + 1

            # Interest on beginning balances
            loan1_int = loan1_bal * (loan1_apr / 12.0)
            loan2_int = loan2_bal * (loan2_apr / 12.0)
            loc_int = loc_bal * loc_r

            # Scheduled principal (simple: IO then amort; business pays all)
            # IO phase: principal 0, payment interest-only
            if month_num <= term_io_months:
                loan1_prin = 0.0
                loan2_prin = 0.0
                personal_ds_m[m] = float(loan2_int)
            else:
                # amort months elapsed since IO ended
                loan1_amort_elapsed = (month_num - term_io_months - 1)
                loan2_amort_elapsed = (month_num - term_io_months - 1)

                loan1_rem = max(0, loan1_amort_after_io - loan1_amort_elapsed) if loan1_bal > 1e-9 else 0
                loan2_rem = max(0, loan2_amort_after_io - loan2_amort_elapsed) if loan2_bal > 1e-9 else 0

                loan1_pmt = loan1_pmt_fixed if loan1_rem > 0 else 0.0
                loan2_pmt = loan2_pmt_fixed if loan2_rem > 0 else 0.0

                loan1_prin = max(0.0, min(loan1_bal, loan1_pmt - loan1_int))
                loan2_prin = max(0.0, min(loan2_bal, loan2_pmt - loan2_int))
                personal_ds_m[m] = float(loan2_int + loan2_prin)

            # Cashflow: business pays all scheduled DS (your requirement)
            debt_service_cash = (loan1_int + loan1_prin) + (loan2_int + loan2_prin) + loc_int
            cash += float(ebitda_cash[m]) - debt_service_cash - other_cash_disbursements_monthly

            # LoC liquidity backstop
            if loc_draw_to_prevent_cash_negative and loc_limit > 0:
                target_cash = float(loc_min_cash_buffer)
                if cash < target_cash:
                    need = target_cash - cash
                    avail = max(0.0, loc_limit - loc_bal)
                    draw = min(need, avail)
                    if draw > 0:
                        loc_bal += draw
                        cash += draw

            # Apply principal to balances (end-of-month)
            loan1_bal = max(0.0, loan1_bal - loan1_prin)
            loan2_bal = max(0.0, loan2_bal - loan2_prin)

            # Auto-paydown LoC from excess cash (optional)
            effective_trust = float(avg_trust_balance) * float(trust_haircut)
            if loc_auto_paydown and loc_bal > 0:
                # keep comp balance floor out of cash before paying down revolver
                if comp_include_personal:
                    comp_base = (loan1_bal + loan2_bal + loc_bal)
                else:
                    comp_base = (loan1_bal + loc_bal)
                required_dep = comp_pct * comp_base
                floor = required_dep  # cash you cannot use if comp is internal; with external trust it still effectively restricts
                excess = max(0.0, cash - floor)
                pay = min(excess, loc_bal)
                if pay > 0:
                    loc_bal -= pay
                    cash -= pay

            # Record cash
            cash_ts[sim, m] = cash

            # Record loan balances (end-of-month, after principal payments and LOC adjustments)
            loan1_bal_ts[sim, m] = loan1_bal
            loan2_bal_ts[sim, m] = loan2_bal
            loc_bal_ts[sim, m] = loc_bal

            # Interest tracking
            # Borrower DSCR always uses BORROWER-ONLY debt (Loan1 + LOC), per lender clarification
            # debt_mode affects Global DSCR and which debt cash is used for, not Borrower DSCR scope
            borrower_entity_interest_m[m] = (loan1_int + loc_int)  # borrower-only interest
            borrower_interest_m[m] = (loan1_int + loc_int)  # ALWAYS borrower-only for Borrower DSCR

            # Funded Debt / EBITDA uses BORROWER-ONLY debt per lender clarification
            # Personal/guarantor debt (Loan2) excluded from this covenant regardless of mode
            funded_debt_bal_m[m] = (loan1_bal + loc_bal)  # ALWAYS borrower-only

            # Compensating balance covenant (monthly)
            if comp_include_personal:
                comp_base = (loan1_bal + loan2_bal + loc_bal)
            else:
                comp_base = (loan1_bal + loc_bal)  # borrower-only comp base
            required_deposit = comp_pct * comp_base
            effective_trust = float(avg_trust_balance) * float(trust_haircut)
            available_for_comp = (cash if comp_balance_cash_eligible else 0.0) + effective_trust
            comp_ok = available_for_comp >= (required_deposit - 1e-6)
            comp_breach_ts[sim, m] = 0.0 if comp_ok else 1.0

            # RLOC rest covenant: LOC must rest at zero for at least one month
            # during each 12-month period.  The borrower can choose any month.
            # We check at the annual boundary (month 12, 24, 36 …) whether the
            # LOC was at zero for any month in the preceding 12-month window.
            if rest_enabled:
                is_annual_boundary = (month_num % 12 == 0)
                if is_annual_boundary:
                    # Look back over the 12-month window (indices m-11 .. m)
                    window_start = max(0, m - 11)
                    rested_any = False
                    for k in range(window_start, m + 1):
                        if loc_bal_ts[sim, k] <= 1e-6:
                            rested_any = True
                            break
                        # Also check current month's end-of-month balance (loc_bal)
                    if loc_bal <= 1e-6:
                        rested_any = True
                    rest_breach_ts[sim, m] = 0.0 if rested_any else 1.0
                else:
                    rest_breach_ts[sim, m] = 0.0
            else:
                rest_breach_ts[sim, m] = 0.0

            # Quarter-end tests
            is_qe = (month_num % 3 == 0)
            is_ae = (month_num % 12 == 0)

            if is_qe:
                # Current maturities: BORROWER-ONLY for Borrower DSCR and D/EBITDA (Loan1 only)
                cm_borrower = scheduled_principal_next12(loan1_bal, loan1_apr, loan1_amort_after_io, term_io_months, m)
                # Note: cm_global includes Loan2 if in 4M mode (computed at annual test)

                curr_maturities_ts[sim, m] = cm_borrower  # store for analytic reverse-solve (borrower scope)
                # Current ratio (uses borrower-only CM per lender clarification)
                cr = current_ratio_value(
                    cash=cash,
                    restricted_cash_balance=restricted_cash_balance,
                    include_restricted_cash_in_ca=include_restricted_cash_in_ca,
                    portfolio_collections_m=port_col,
                    m=m,
                    other_ca=other_ca,
                    other_cl=other_cl,
                    portfolio_current_months=portfolio_current_months,
                    portfolio_current_haircut=portfolio_current_haircut,
                    include_rloc_in_cl=include_rloc_in_cl,
                    rloc_balance=loc_bal,
                    curr_maturities=cm_borrower
                )
                cr_val_ts[sim, m] = cr
                cr_breach_ts[sim, m] = 1.0 if (np.isfinite(cr) and cr < current_ratio_min) else 0.0

                # DSCR (borrower) — term sheet excludes IO period from testing
                if month_num > term_io_months:
                    ds = borrower_dscr_value(
                        ebitda_m=ebitda,
                        borrower_interest_m=borrower_interest_m,
                        curr_maturities=cm_borrower,
                        m=m,
                        A=A
                    )
                    dscr_val_ts[sim, m] = ds
                    dscr_breach_ts[sim, m] = 1.0 if (np.isfinite(ds) and ds < dscr_min) else 0.0

                # Debt / EBITDA (TTM) - only test once IO complete and 12 months exist
                if month_num > term_io_months and month_num >= 12:
                    dte = debt_to_ebitda_value(ebitda, funded_debt_bal_m, m)
                    dte_val_ts[sim, m] = dte
                    dmax = dte_max_12 if month_num <= 12 else dte_max_after
                    dte_breach_ts[sim, m] = 1.0 if (np.isfinite(dte) and dte > dmax) else 0.0

            # Annual test
            if is_ae and month_num >= 12:
                # Global DSCR per term sheet:
                # Den = (CM + Interest) + (Personal P&I)
                # Since personal_ds_m includes Loan2 full P&I (interest + principal),
                # CM here should be BORROWER-ONLY to avoid double-counting Loan2 principal.
                # Interest portion: borrower_interest_m is Loan1+LOC; personal_ds_m has Loan2 interest.
                cm_for_gdscr = scheduled_principal_next12(loan1_bal, loan1_apr, loan1_amort_after_io, term_io_months, m)
                # Note: Loan2 CM is NOT added here because it's in personal_ds_m as part of P&I

                gd = global_dscr_value(
                    ebitda_m=ebitda,
                    borrower_interest_m=borrower_interest_m,
                    curr_maturities=cm_for_gdscr,
                    personal_ds_m=personal_ds_m,
                    m=m,
                    debt_mode=debt_mode,
                    A=A
                )
                gdscr_val_ts[sim, m] = gd
                gdscr_breach_ts[sim, m] = 1.0 if (np.isfinite(gd) and gd < gdscr_min) else 0.0

        # After monthly loop: record the per-sim arrays to output arrays
        borrower_interest_ts[sim, :] = borrower_interest_m
        personal_ds_ts[sim, :] = personal_ds_m

    # "Ever" source of truth
    ever_cash_neg = (np.any(cash_ts < 0, axis=1)).astype(int)
    ever_ebitda_neg = (np.any(ebitda_ts < 0, axis=1)).astype(int)

    def ever_breach(ts: np.ndarray) -> float:
        tested = np.isfinite(ts)
        breached = (ts > 0.5) & tested
        return float(np.mean(np.any(breached, axis=1)))

    # Build full-horizon base curve (zero-padded beyond curve_h)
    base_curve_full = np.zeros(horizon, dtype=float)
    base_curve_full[:curve_h] = base_w
    expected_collections_full = base_total_col * base_curve_full

    return {
        "assumptions": A,
        "debt_mode": debt_mode,
        "n_sims": n_sims,
        "horizon_months": horizon,
        "checkpoint_months": checkpoint_months,

        # Curve data for analysis
        "base_curve_weights": base_curve_full,
        "base_total_collections": base_total_col,
        "expected_collections": expected_collections_full,
        "curve_horizon": curve_h,
        "collections_sigma": col_cv,

        # New client revenue overlay
        "new_client_enabled": new_client_enabled,
        "new_client_base_m": new_client_base_m,

        "cash_ts": cash_ts,
        "ebitda_ts": ebitda_ts,
        "revenue_ts": revenue_ts,
        "opex_ts": opex_ts,
        "portcol_ts": portcol_ts,
        "borrower_interest_ts": borrower_interest_ts,
        "personal_ds_ts": personal_ds_ts,
        "funded_debt_bal_ts": funded_debt_bal_ts,
        "curr_maturities_ts": curr_maturities_ts,
        "cr_val_ts": cr_val_ts,
        "dscr_val_ts": dscr_val_ts,
        "dte_val_ts": dte_val_ts,
        "gdscr_val_ts": gdscr_val_ts,

        # Loan balance time series
        "loan1_bal_ts": loan1_bal_ts,
        "loan2_bal_ts": loan2_bal_ts,
        "loc_bal_ts": loc_bal_ts,

        "comp_breach_ts": comp_breach_ts,
        "cr_breach_ts": cr_breach_ts,
        "dscr_breach_ts": dscr_breach_ts,
        "dte_breach_ts": dte_breach_ts,
        "gdscr_breach_ts": gdscr_breach_ts,
        "rest_breach_ts": rest_breach_ts,

        "fail_comp_pct": ever_breach(comp_breach_ts),
        "fail_current_ratio_pct": ever_breach(cr_breach_ts),
        "fail_dscr_pct": ever_breach(dscr_breach_ts),
        "fail_debt_to_ebitda_pct": ever_breach(dte_breach_ts),
        "fail_gdscr_pct": ever_breach(gdscr_breach_ts),
        "fail_rloc_rest_pct": ever_breach(rest_breach_ts),
        "fail_cash_negative_ever_pct": float(ever_cash_neg.mean()),
        "fail_ebitda_negative_ever_pct": float(ever_ebitda_neg.mean()),
    }

# ============================================================
# Reporting
# ============================================================
def print_ever_breach_summary(r: Dict) -> None:
    A = r["assumptions"]
    print("---- MONTE CARLO COVENANT SUMMARY ----")
    print(f"Mode: {r['debt_mode']} | Simulations: {r['n_sims']:,} | Horizon: {r['horizon_months']} months\n")
    print("Breach probability (ever at any tested period):")
    print(f"  Compensating balance:      {fmt_pct(r['fail_comp_pct'])}")
    print(f"  Current ratio (>=min):     {fmt_pct(r['fail_current_ratio_pct'])}")
    print(f"  DSCR (>=min):              {fmt_pct(r['fail_dscr_pct'])}")
    print(f"  Funded Debt / EBITDA:      {fmt_pct(r['fail_debt_to_ebitda_pct'])}")
    print(f"  Global DSCR (>=min):       {fmt_pct(r['fail_gdscr_pct'])}")
    print(f"  RLOC annual rest:          {fmt_pct(r['fail_rloc_rest_pct'])}")
    print(f"  Cash < 0 (ever):           {fmt_pct(r['fail_cash_negative_ever_pct'])}")
    print(f"  EBITDA < 0 (ever):         {fmt_pct(r['fail_ebitda_negative_ever_pct'])}")
    print("-------------------------------------\n")

def checkpoint_breach_table(r: Dict) -> pd.DataFrame:
    """
    CUMULATIVE breach probability through each checkpoint month.
    Non-test months are ignored (NaN).
    """
    def p_breach_through(ts_name: str, idx: int) -> float:
        window = r[ts_name][:, :idx + 1]
        tested = np.isfinite(window)
        if not np.any(tested):
            return np.nan
        breached = (window > 0.5) & tested
        return float(np.mean(np.any(breached, axis=1)))

    def p_cash_neg_through(idx: int) -> float:
        win = r["cash_ts"][:, :idx + 1]
        return float(np.mean(np.any(win < 0, axis=1)))

    def p_ebitda_neg_through(idx: int) -> float:
        win = r["ebitda_ts"][:, :idx + 1]
        return float(np.mean(np.any(win < 0, axis=1)))

    rows = []
    print("---- CHECKPOINT BREACH PROBABILITIES (CUMULATIVE THROUGH MONTH) ----")
    for mth in r["checkpoint_months"]:
        idx = mth - 1
        row = {
            "Month": mth,
            "P(Comp Bal breach)": p_breach_through("comp_breach_ts", idx),
            "P(Current Ratio breach)": p_breach_through("cr_breach_ts", idx),
            "P(DSCR breach)": p_breach_through("dscr_breach_ts", idx),
            "P(Debt/EBITDA breach)": p_breach_through("dte_breach_ts", idx),
            "P(Global DSCR breach)": p_breach_through("gdscr_breach_ts", idx),
            "P(RLOC rest breach)": p_breach_through("rest_breach_ts", idx),
            "P(Cash < 0)": p_cash_neg_through(idx),
            "P(EBITDA < 0)": p_ebitda_neg_through(idx),
        }
        rows.append(row)

        print(f"\n===== MONTH {mth} =====")
        print(f"  Compensating balance:      {fmt_pct(row['P(Comp Bal breach)'])}")
        print(f"  Current ratio (>=min):     {fmt_pct(row['P(Current Ratio breach)'])}")
        print(f"  DSCR (>=min):              {fmt_pct(row['P(DSCR breach)'])}")
        print(f"  Funded Debt / EBITDA:      {fmt_pct(row['P(Debt/EBITDA breach)'])}")
        print(f"  Global DSCR (>=min):       {fmt_pct(row['P(Global DSCR breach)'])}")
        print(f"  RLOC annual rest:          {fmt_pct(row['P(RLOC rest breach)'])}")
        print(f"  Cash < 0 (through month):  {fmt_pct(row['P(Cash < 0)'])}")
        print(f"  EBITDA < 0 (through month):{fmt_pct(row['P(EBITDA < 0)'])}")

    print("\n----------------------------------------\n")
    return pd.DataFrame(rows)

def checkpoint_snapshots_table(r: Dict) -> pd.DataFrame:
    """
    Point-in-time distribution stats at each checkpoint month.
    Notes:
      - P50 is the median (50th percentile).
      - P5 is the 5th percentile (a downside tail outcome).
    """
    A = r["assumptions"]
    dscr_min = f(A, "dscr_min", 1.30)
    gdscr_min = f(A, "gdscr_min", 2.0)
    dte_max_12 = f(A, "debt_to_ebitda_max_first12", 2.25)
    dte_max_after = f(A, "debt_to_ebitda_max_after12", 1.75)

    rows = []
    print("---- CHECKPOINT SNAPSHOTS (POINT-IN-TIME) ----")
    for mth in r["checkpoint_months"]:
        idx = mth - 1
        cash = r["cash_ts"][:, idx]
        ebitda = r["ebitda_ts"][:, idx]
        dscr = r["dscr_val_ts"][:, idx]
        dte = r["dte_val_ts"][:, idx]
        gdscr = r["gdscr_val_ts"][:, idx]
        dmax = dte_max_12 if mth <= 12 else dte_max_after

        row = {
            "Month": mth,
            "Cash_P50_median": safe_median(cash),
            "Cash_P5": safe_percentile(cash, 5),
            "P(Cash<0)": safe_prob(cash, lambda x: x < 0),

            "EBITDA_P50_median": safe_median(ebitda),
            "EBITDA_P5": safe_percentile(ebitda, 5),
            "P(EBITDA<0)": safe_prob(ebitda, lambda x: x < 0),

            "DSCR_P50_median": safe_median(dscr),
            "DSCR_P5": safe_percentile(dscr, 5),
            "P(DSCR<min)": safe_prob(dscr, lambda x: x < dscr_min),

            "DTE_P50_median": safe_median(dte),
            "DTE_P95": safe_percentile(dte, 95),
            "P(DTE>max)": safe_prob(dte, lambda x: x > dmax),

            "GDSCR_P50_median": safe_median(gdscr),
            "GDSCR_P5": safe_percentile(gdscr, 5),
            "P(GDSCR<min)": safe_prob(gdscr, lambda x: x < gdscr_min),
        }
        rows.append(row)

        print(f"\n===== MONTH {mth} =====")
        print(f"Cash:   Median(P50) {fmt_money(row['Cash_P50_median'])} | P5 {fmt_money(row['Cash_P5'])} | P(Cash<0) {fmt_pct(row['P(Cash<0)'])}")
        print(f"EBITDA: Median(P50) {fmt_money(row['EBITDA_P50_median'])} | P5 {fmt_money(row['EBITDA_P5'])} | P(EBITDA<0) {fmt_pct(row['P(EBITDA<0)'])}")
        print(f"DSCR:   Median(P50) {fmt_num(row['DSCR_P50_median'])} | P5 {fmt_num(row['DSCR_P5'])} | P(DSCR<min) {fmt_pct(row['P(DSCR<min)'])}")
        print(f"D/EBITDA: Median(P50) {fmt_num(row['DTE_P50_median'])}x | P95 {fmt_num(row['DTE_P95'])}x | P(>max) {fmt_pct(row['P(DTE>max)'])}")
        print(f"GDSCR:  Median(P50) {fmt_num(row['GDSCR_P50_median'])} | P5 {fmt_num(row['GDSCR_P5'])} | P(GDSCR<min) {fmt_pct(row['P(GDSCR<min)'])}")

    print("\n---------------------------------------------\n")
    return pd.DataFrame(rows)

# ============================================================
# Fan charts (optional)
# ============================================================
def fan_bands(ts: np.ndarray, ps=(5, 25, 50, 75, 95)) -> Dict[int, np.ndarray]:
    ts = np.asarray(ts, dtype=float)
    horizon = ts.shape[1]
    out = {p: np.full(horizon, np.nan, dtype=float) for p in ps}
    for t in range(horizon):
        col = _finite(ts[:, t])
        if col.size == 0:
            continue
        for p in ps:
            out[p][t] = np.percentile(col, p)
    return out

def plot_fan(ts: np.ndarray, title: str, ylabel: str) -> None:
    bands = fan_bands(ts)
    x = np.arange(1, ts.shape[1] + 1)
    plt.figure()
    plt.fill_between(x, bands[5], bands[95], alpha=0.2)
    plt.fill_between(x, bands[25], bands[75], alpha=0.35)
    plt.plot(x, bands[50])
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.2)

def plot_all_fans(r: Dict) -> None:
    plot_fan(r["cash_ts"], "Cash fan chart (P5/P25/Median/P75/P95)", "Cash")
    plot_fan(r["ebitda_ts"], "EBITDA fan chart (P5/P25/Median/P75/P95)", "EBITDA (monthly)")
    plot_fan(r["dscr_val_ts"], "DSCR fan chart (quarter-end values)", "DSCR")
    plot_fan(r["dte_val_ts"], "Debt/EBITDA fan chart (quarter-end values)", "Debt/EBITDA")
    plt.show()

# ============================================================
# Analytic reverse-solve: required rev uplift / expense cut
# ============================================================
def required_change_by_checkpoint_analytic(r: Dict,
                                          covenant: str,
                                          mode: str,
                                          target_prob: float = 0.15) -> Dict[int, float]:
    """
    Fast solve for the required revenue uplift or expense cut so that cumulative breach probability
    through each checkpoint is <= target_prob.

    IMPORTANT: This respects actual covenant test timing:
      - DTE (Funded Debt/EBITDA): quarter-end only, and only once 12 months exist (month>=12).
      - GDSCR (Global DSCR): annual only (months 12,24,36,...).

    It also respects debt_mode:
      - 2M: GDSCR excludes personal debt service and personal cash flow.
      - 4M: GDSCR includes simulated personal debt service (personal_ds_ts) and optional add-backs.
    """
    covenant = covenant.upper()
    mode = mode.upper()
    if covenant not in {"DTE", "GDSCR"}:
        raise ValueError("covenant must be 'DTE' or 'GDSCR'")
    if mode not in {"REVENUE", "EXPENSES"}:
        raise ValueError("mode must be 'REVENUE' or 'EXPENSES'")
    target_prob = float(target_prob)
    if not (0.0 < target_prob < 1.0):
        raise ValueError("target_prob must be between 0 and 1")

    A = r["assumptions"]
    debt_mode = str(r.get("debt_mode", s(A, "debt_mode", "2M"))).upper()
    checkpoints = r["checkpoint_months"]
    horizon = int(r["horizon_months"])
    n_sims = int(r["n_sims"])

    ebitda_ts = np.asarray(r["ebitda_ts"], dtype=float)
    revenue_ts = np.asarray(r["revenue_ts"], dtype=float)
    opex_ts = np.asarray(r["opex_ts"], dtype=float)
    borrower_interest_ts = np.asarray(r["borrower_interest_ts"], dtype=float)
    funded_debt_bal_ts = np.asarray(r["funded_debt_bal_ts"], dtype=float)
    curr_maturities_ts = np.asarray(r["curr_maturities_ts"], dtype=float)

    # optional (present in v7+)
    personal_ds_ts = np.asarray(r.get("personal_ds_ts", np.zeros_like(ebitda_ts)), dtype=float)

    # Covenant thresholds
    gdscr_min = f(A, "gdscr_min", 2.0)
    dte_max_12 = f(A, "debt_to_ebitda_max_first12", 2.25)
    dte_max_after = f(A, "debt_to_ebitda_max_after12", 1.75)

    # Annual adjustments (treated as flat, borrower-friendly; consistent with model)
    taxes_m = f(A, "cash_taxes_annual", 0.0) / 12.0
    div_m = f(A, "dividends_annual", 0.0) / 12.0
    dist_m = f(A, "distributions_annual", 0.0) / 12.0

    personal_cf_annual = f(A, "personal_cash_flow_annual", 0.0)
    extra_personal_ds_annual = f(A, "extra_personal_ds_annual", 0.0)

    # helper trailing sum matrix
    def ttm(mat: np.ndarray) -> np.ndarray:
        out = np.full_like(mat, np.nan, dtype=float)
        for t in range(mat.shape[1]):
            start = max(0, t - 11)
            out[:, t] = np.sum(mat[:, start:t+1], axis=1)
        return out

    ebitda_ttm = ttm(ebitda_ts)
    rev_ttm = ttm(revenue_ts)
    opex_ttm = ttm(opex_ts)
    adj_ebitda_ts = ebitda_ts - (taxes_m + div_m + dist_m)
    adj_ttm = ttm(adj_ebitda_ts)
    int_ttm = ttm(borrower_interest_ts)
    personal_ds_ttm = ttm(personal_ds_ts)

    # Determine test months mask per covenant
    months = np.arange(horizon, dtype=int)  # 0-index
    qe_mask = ((months + 1) % 3 == 0) & (months >= 11)
    ae_mask = ((months + 1) % 12 == 0) & (months >= 11)

    # Precompute per-month required pct per sim
    req = np.zeros((n_sims, horizon), dtype=float)

    if covenant == "DTE":
        # funded_debt / EBITDA <= max
        for t in range(horizon):
            if not qe_mask[t]:
                continue
            dmax = dte_max_12 if (t + 1) <= 12 else dte_max_after
            # Need EBITDA_ttm >= funded_debt / dmax
            needed_ebitda = funded_debt_bal_ts[:, t] / max(dmax, 1e-12)
            gap = needed_ebitda - ebitda_ttm[:, t]
            gap = np.clip(gap, 0.0, None)
            denom = rev_ttm[:, t] if mode == "REVENUE" else opex_ttm[:, t]
            denom = np.where(np.abs(denom) < 1e-9, np.nan, denom)
            req[:, t] = gap / denom
            req[:, t] = np.where(np.isfinite(req[:, t]), np.clip(req[:, t], 0.0, None), np.inf)

    else:  # GDSCR
        # (Adj EBITDA TTM + personal_cf_annual [4M]) / (Int TTM + CM + personal_ds_TTM [4M] + extra_personal_ds_annual [4M]) >= min
        for t in range(horizon):
            if not ae_mask[t]:
                continue
            num = adj_ttm[:, t].copy()
            den = int_ttm[:, t] + curr_maturities_ts[:, t]
            if debt_mode == "4M":
                num = num + float(personal_cf_annual)
                den = den + personal_ds_ttm[:, t] + float(extra_personal_ds_annual)
            # Required increase in numerator via revenue uplift or expense cut:
            # num + pct * denom_component >= gmin * den
            gap = (gdscr_min * den) - num
            gap = np.clip(gap, 0.0, None)
            denom_comp = rev_ttm[:, t] if mode == "REVENUE" else opex_ttm[:, t]
            denom_comp = np.where(np.abs(denom_comp) < 1e-9, np.nan, denom_comp)
            req[:, t] = gap / denom_comp
            req[:, t] = np.where(np.isfinite(req[:, t]), np.clip(req[:, t], 0.0, None), np.inf)

    # Now convert to cumulative through checkpoints:
    out = {}
    for chk in checkpoints:
        tmax = min(int(chk) - 1, horizon - 1)
        # consider only months <= checkpoint that are test months for this covenant
        mask = qe_mask if covenant == "DTE" else ae_mask
        eligible = mask.copy()
        eligible[tmax+1:] = False
        if not np.any(eligible):
            out[int(chk)] = 0.0
            continue
        per_sim_req = np.max(req[:, eligible], axis=1)
        # (1-target) percentile of required pct gives pct so that <=target exceed it
        pct_needed = float(np.percentile(per_sim_req, 100.0 * (1.0 - target_prob)))
        if not np.isfinite(pct_needed):
            pct_needed = 0.0
        out[int(chk)] = max(0.0, pct_needed)
    return out

# ============================================================
# Monthly Balance Summary
# ============================================================
def monthly_balance_summary(r: Dict) -> pd.DataFrame:
    """
    Generate monthly summary of Cash, LOC, Loan1, Loan2 balances.
    Returns percentiles (P0/min, P5, P10, P25, P50/median, P75, P90, P95, P100/max).
    """
    horizon = r["horizon_months"]
    
    metrics = {
        "Cash": r["cash_ts"],
        "LOC_Balance": r["loc_bal_ts"],
        "Loan1_Balance": r["loan1_bal_ts"],
        "Loan2_Balance": r["loan2_bal_ts"],
        "EBITDA": r["ebitda_ts"],
    }
    
    rows = []
    for m in range(horizon):
        month_num = m + 1
        row = {"Month": month_num}
        
        for name, ts in metrics.items():
            col = ts[:, m]
            finite = col[np.isfinite(col)]
            if len(finite) == 0:
                for p in ["P0", "P5", "P10", "P25", "P50", "P75", "P90", "P95", "P100"]:
                    row[f"{name}_{p}"] = np.nan
            else:
                row[f"{name}_P0"] = np.min(finite)
                row[f"{name}_P5"] = np.percentile(finite, 5)
                row[f"{name}_P10"] = np.percentile(finite, 10)
                row[f"{name}_P25"] = np.percentile(finite, 25)
                row[f"{name}_P50"] = np.percentile(finite, 50)
                row[f"{name}_P75"] = np.percentile(finite, 75)
                row[f"{name}_P90"] = np.percentile(finite, 90)
                row[f"{name}_P95"] = np.percentile(finite, 95)
                row[f"{name}_P100"] = np.max(finite)
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def print_monthly_balance_summary(r: Dict, months: Optional[List[int]] = None) -> pd.DataFrame:
    """Print and return monthly balance summary for selected months."""
    df = monthly_balance_summary(r)
    
    if months is None:
        # Default to quarterly months
        months = [m for m in range(3, r["horizon_months"] + 1, 3)]
    
    print("\n---- MONTHLY BALANCE SUMMARY (Selected Months) ----")
    print("Values shown: P5 / P50 (Median) / P95\n")
    
    for m in months:
        if m > r["horizon_months"]:
            continue
        row = df[df["Month"] == m].iloc[0]
        print(f"===== MONTH {m} =====")
        print(f"  Cash:        ${row['Cash_P5']:>12,.0f} / ${row['Cash_P50']:>12,.0f} / ${row['Cash_P95']:>12,.0f}")
        print(f"  LOC Balance: ${row['LOC_Balance_P5']:>12,.0f} / ${row['LOC_Balance_P50']:>12,.0f} / ${row['LOC_Balance_P95']:>12,.0f}")
        print(f"  Loan1 Bal:   ${row['Loan1_Balance_P5']:>12,.0f} / ${row['Loan1_Balance_P50']:>12,.0f} / ${row['Loan1_Balance_P95']:>12,.0f}")
        print(f"  Loan2 Bal:   ${row['Loan2_Balance_P5']:>12,.0f} / ${row['Loan2_Balance_P50']:>12,.0f} / ${row['Loan2_Balance_P95']:>12,.0f}")
        print()
    
    return df

# ============================================================
# Required Revenue/Expense Change Analysis (Multi-Target)
# ============================================================
def required_changes_multi_target(r: Dict, 
                                   targets: List[float] = [0.15, 0.10, 0.05, 0.00]) -> Dict[str, pd.DataFrame]:
    """
    Compute required revenue uplift and expense cuts for multiple target breach probabilities.
    
    Returns dict with keys:
        'DTE_revenue', 'DTE_expenses', 'GDSCR_revenue', 'GDSCR_expenses'
    Each value is a DataFrame with columns: Month, P15, P10, P5, P0 (or whatever targets specified)
    """
    A = r["assumptions"]
    
    # For GAAP basis, we use the actual mean values from the simulation
    # For dollar conversion, approximate from actuals
    rev_mean = np.nanmean(r["revenue_ts"][:, 0])  # First month mean revenue
    exp_mean = np.nanmean(r["opex_ts"][:, 0])     # First month mean expenses
    
    results = {}
    
    for covenant in ["DTE", "GDSCR"]:
        for mode in ["REVENUE", "EXPENSES"]:
            key = f"{covenant}_{mode.lower()}"
            rows = []
            
            for mth in r["checkpoint_months"]:
                row = {"Month": mth}
                for target in targets:
                    # Handle P0 (target=0.0) by using a tiny positive number
                    t = max(target, 0.001)
                    try:
                        req = required_change_by_checkpoint_analytic(r, covenant=covenant, mode=mode, target_prob=t)
                        pct = req.get(mth, np.nan)
                    except:
                        pct = np.nan
                    
                    target_label = f"P{int(target*100)}"
                    row[f"{target_label}_pct"] = pct
                    
                    # Convert to dollars
                    if mode == "REVENUE":
                        row[f"{target_label}_dollars"] = rev_mean * pct if np.isfinite(pct) else np.nan
                    else:
                        row[f"{target_label}_dollars"] = exp_mean * pct if np.isfinite(pct) else np.nan
                
                rows.append(row)
            
            results[key] = pd.DataFrame(rows)
    
    return results

def print_required_changes_summary(r: Dict, 
                                    targets: List[float] = [0.15, 0.10, 0.05, 0.00]) -> Dict[str, pd.DataFrame]:
    """Print and return required revenue/expense changes for covenant cure."""
    
    results = required_changes_multi_target(r, targets)
    
    print("\n" + "="*80)
    print("REQUIRED REVENUE/EXPENSE CHANGES FOR COVENANT CURE")
    print("="*80)
    print("Shows monthly revenue increase OR expense decrease needed to achieve target breach probability")
    print()
    
    for covenant in ["DTE", "GDSCR"]:
        covenant_name = "Funded Debt / EBITDA" if covenant == "DTE" else "Global DSCR"
        print(f"\n---- {covenant_name} ----")
        
        for mode in ["REVENUE", "EXPENSES"]:
            key = f"{covenant}_{mode.lower()}"
            df = results[key]
            
            action = "Revenue Increase" if mode == "REVENUE" else "Expense Decrease"
            print(f"\n{action} Required:")
            print("-" * 70)
            
            # Header
            header = f"{'Month':>6}"
            for target in targets:
                label = f"P{int(target*100)}"
                header += f" | {label:>18}"
            print(header)
            print("-" * 70)
            
            for _, row in df.iterrows():
                line = f"{int(row['Month']):>6}"
                for target in targets:
                    label = f"P{int(target*100)}"
                    pct = row.get(f"{label}_pct", np.nan)
                    dollars = row.get(f"{label}_dollars", np.nan)
                    if np.isfinite(pct) and np.isfinite(dollars):
                        line += f" | {pct*100:>5.1f}% (${dollars:>8,.0f})"
                    else:
                        line += f" | {'N/A':>18}"
                print(line)
    
    print("\n" + "="*80)
    return results

# ============================================================
# Excel Export (Comprehensive Data Output)
# ============================================================
def export_results_to_excel(r: Dict, 
                            filepath: str = "mc_output_data.xlsx",
                            include_cure_analysis: bool = True,
                            cure_targets: List[float] = [0.15, 0.10, 0.05, 0.00]) -> str:
    """
    Export all simulation results to an Excel file structured for linking.
    
    This file is designed to be the single data source that presentation
    workbooks (Monte_Carlo_Loan_Model_Regression.xlsx, Covenant_And_Cash_Results.xlsx)
    can link to via Excel formulas.
    
    Sheets:
        - Executive_Data: Key metrics in tabular format for executive summary linking
        - Breach_Cumulative: Cumulative breach probabilities by checkpoint
        - Breach_PointInTime: Point-in-time breach probabilities at each checkpoint
        - Snapshots: Point-in-time distributions (P5/P50/P95) at checkpoints
        - Monthly_Cash: Full monthly cash balance percentile distribution
        - Monthly_EBITDA: Full monthly EBITDA percentile distribution
        - Monthly_Debt: Loan1, Loan2, LOC balances by month
        - Monthly_Covenants: DSCR, DTE, GDSCR values by month (at test months)
        - Cure_GDSCR: Required revenue/expense changes for Global DSCR cure
        - Cure_DTE: Required revenue/expense changes for D/EBITDA cure
        - Assumptions: Input assumptions echo
        - Run_Info: Timestamp and run metadata
    
    Returns: filepath
    """
    from datetime import datetime
    
    A = r["assumptions"]
    horizon = r["horizon_months"]
    n_sims = r["n_sims"]
    
    # Covenant thresholds
    dscr_min = f(A, "dscr_min", 1.30)
    gdscr_min = f(A, "gdscr_min", 2.0)
    dte_max_12 = f(A, "debt_to_ebitda_max_first12", 2.25)
    dte_max_after = f(A, "debt_to_ebitda_max_after12", 1.75)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        
        # ================================================================
        # 1. EXECUTIVE DATA - Key metrics in linkable tabular format
        # ================================================================
        exec_rows = [
            {"Category": "Configuration", "Metric": "Debt Mode", "Value": r["debt_mode"], "Value_Numeric": np.nan},
            {"Category": "Configuration", "Metric": "Simulations", "Value": str(r["n_sims"]), "Value_Numeric": float(r["n_sims"])},
            {"Category": "Configuration", "Metric": "Horizon Months", "Value": str(r["horizon_months"]), "Value_Numeric": float(r["horizon_months"])},
            {"Category": "Configuration", "Metric": "Run Timestamp", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Value_Numeric": np.nan},
            {"Category": "Breach Prob (Ever)", "Metric": "Compensating Balance", "Value": f"{r['fail_comp_pct']*100:.2f}%", "Value_Numeric": r['fail_comp_pct']},
            {"Category": "Breach Prob (Ever)", "Metric": "Current Ratio", "Value": f"{r['fail_current_ratio_pct']*100:.2f}%", "Value_Numeric": r['fail_current_ratio_pct']},
            {"Category": "Breach Prob (Ever)", "Metric": "Borrower DSCR", "Value": f"{r['fail_dscr_pct']*100:.2f}%", "Value_Numeric": r['fail_dscr_pct']},
            {"Category": "Breach Prob (Ever)", "Metric": "Funded Debt / EBITDA", "Value": f"{r['fail_debt_to_ebitda_pct']*100:.2f}%", "Value_Numeric": r['fail_debt_to_ebitda_pct']},
            {"Category": "Breach Prob (Ever)", "Metric": "Global DSCR", "Value": f"{r['fail_gdscr_pct']*100:.2f}%", "Value_Numeric": r['fail_gdscr_pct']},
            {"Category": "Breach Prob (Ever)", "Metric": "RLOC Annual Rest", "Value": f"{r['fail_rloc_rest_pct']*100:.2f}%", "Value_Numeric": r['fail_rloc_rest_pct']},
            {"Category": "Breach Prob (Ever)", "Metric": "Cash < 0", "Value": f"{r['fail_cash_negative_ever_pct']*100:.2f}%", "Value_Numeric": r['fail_cash_negative_ever_pct']},
            {"Category": "Breach Prob (Ever)", "Metric": "EBITDA < 0", "Value": f"{r['fail_ebitda_negative_ever_pct']*100:.2f}%", "Value_Numeric": r['fail_ebitda_negative_ever_pct']},
            {"Category": "Thresholds", "Metric": "DSCR Minimum", "Value": str(dscr_min), "Value_Numeric": dscr_min},
            {"Category": "Thresholds", "Metric": "GDSCR Minimum", "Value": str(gdscr_min), "Value_Numeric": gdscr_min},
            {"Category": "Thresholds", "Metric": "D/EBITDA Max (First 12mo)", "Value": str(dte_max_12), "Value_Numeric": dte_max_12},
            {"Category": "Thresholds", "Metric": "D/EBITDA Max (After 12mo)", "Value": str(dte_max_after), "Value_Numeric": dte_max_after},
        ]
        pd.DataFrame(exec_rows).to_excel(writer, sheet_name="Executive_Data", index=False)
        
        # ================================================================
        # 2. BREACH_CUMULATIVE - Cumulative breach probabilities by checkpoint
        # ================================================================
        breach_rows = []
        for mth in r["checkpoint_months"]:
            idx = mth - 1
            def p_breach_through(ts_name: str) -> float:
                window = r[ts_name][:, :idx + 1]
                tested = np.isfinite(window)
                if not np.any(tested):
                    return 0.0
                breached = (window > 0.5) & tested
                return float(np.mean(np.any(breached, axis=1)))
            
            breach_rows.append({
                "Month": mth,
                "Comp_Balance": p_breach_through("comp_breach_ts"),
                "Current_Ratio": p_breach_through("cr_breach_ts"),
                "Borrower_DSCR": p_breach_through("dscr_breach_ts"),
                "Debt_to_EBITDA": p_breach_through("dte_breach_ts"),
                "Global_DSCR": p_breach_through("gdscr_breach_ts"),
                "RLOC_Rest": p_breach_through("rest_breach_ts"),
                "Cash_Negative": float(np.mean(np.any(r["cash_ts"][:, :idx+1] < 0, axis=1))),
                "EBITDA_Negative": float(np.mean(np.any(r["ebitda_ts"][:, :idx+1] < 0, axis=1))),
            })
        pd.DataFrame(breach_rows).to_excel(writer, sheet_name="Breach_Cumulative", index=False)
        
        # ================================================================
        # 3. BREACH_POINTINTIME - Point-in-time breach at each checkpoint
        # ================================================================
        pit_breach_rows = []
        for mth in r["checkpoint_months"]:
            idx = mth - 1
            def p_breach_at(ts_name: str) -> float:
                val = r[ts_name][:, idx]
                tested = np.isfinite(val)
                if not np.any(tested):
                    return 0.0
                breached = (val > 0.5) & tested
                return float(np.mean(breached))
            
            pit_breach_rows.append({
                "Month": mth,
                "Comp_Balance": p_breach_at("comp_breach_ts"),
                "Current_Ratio": p_breach_at("cr_breach_ts"),
                "Borrower_DSCR": p_breach_at("dscr_breach_ts"),
                "Debt_to_EBITDA": p_breach_at("dte_breach_ts"),
                "Global_DSCR": p_breach_at("gdscr_breach_ts"),
                "RLOC_Rest": p_breach_at("rest_breach_ts"),
                "Cash_Negative": float(np.mean(r["cash_ts"][:, idx] < 0)),
                "EBITDA_Negative": float(np.mean(r["ebitda_ts"][:, idx] < 0)),
            })
        pd.DataFrame(pit_breach_rows).to_excel(writer, sheet_name="Breach_PointInTime", index=False)
        
        # ================================================================
        # 4. SNAPSHOTS - Point-in-time distributions at checkpoints
        # ================================================================
        snapshot_rows = []
        for mth in r["checkpoint_months"]:
            idx = mth - 1
            cash = r["cash_ts"][:, idx]
            ebitda = r["ebitda_ts"][:, idx]
            dscr = r["dscr_val_ts"][:, idx]
            dte = r["dte_val_ts"][:, idx]
            gdscr = r["gdscr_val_ts"][:, idx]
            
            snapshot_rows.append({
                "Month": mth,
                "Cash_P5": safe_percentile(cash, 5),
                "Cash_P25": safe_percentile(cash, 25),
                "Cash_P50": safe_median(cash),
                "Cash_P75": safe_percentile(cash, 75),
                "Cash_P95": safe_percentile(cash, 95),
                "EBITDA_P5": safe_percentile(ebitda, 5),
                "EBITDA_P25": safe_percentile(ebitda, 25),
                "EBITDA_P50": safe_median(ebitda),
                "EBITDA_P75": safe_percentile(ebitda, 75),
                "EBITDA_P95": safe_percentile(ebitda, 95),
                "DSCR_P5": safe_percentile(dscr, 5),
                "DSCR_P25": safe_percentile(dscr, 25),
                "DSCR_P50": safe_median(dscr),
                "DSCR_P75": safe_percentile(dscr, 75),
                "DSCR_P95": safe_percentile(dscr, 95),
                "DTE_P5": safe_percentile(dte, 5),
                "DTE_P25": safe_percentile(dte, 25),
                "DTE_P50": safe_median(dte),
                "DTE_P75": safe_percentile(dte, 75),
                "DTE_P95": safe_percentile(dte, 95),
                "GDSCR_P5": safe_percentile(gdscr, 5),
                "GDSCR_P25": safe_percentile(gdscr, 25),
                "GDSCR_P50": safe_median(gdscr),
                "GDSCR_P75": safe_percentile(gdscr, 75),
                "GDSCR_P95": safe_percentile(gdscr, 95),
            })
        pd.DataFrame(snapshot_rows).to_excel(writer, sheet_name="Snapshots", index=False)
        
        # ================================================================
        # 5. MONTHLY_CASH - Full cash balance time series
        # ================================================================
        cash_rows = []
        for m in range(horizon):
            col = r["cash_ts"][:, m]
            finite = col[np.isfinite(col)]
            if len(finite) == 0:
                cash_rows.append({"Month": m+1, "P0": np.nan, "P5": np.nan, "P10": np.nan, 
                                  "P25": np.nan, "P50": np.nan, "P75": np.nan, 
                                  "P90": np.nan, "P95": np.nan, "P100": np.nan, "Mean": np.nan})
            else:
                cash_rows.append({
                    "Month": m + 1,
                    "P0": np.min(finite),
                    "P5": np.percentile(finite, 5),
                    "P10": np.percentile(finite, 10),
                    "P25": np.percentile(finite, 25),
                    "P50": np.percentile(finite, 50),
                    "P75": np.percentile(finite, 75),
                    "P90": np.percentile(finite, 90),
                    "P95": np.percentile(finite, 95),
                    "P100": np.max(finite),
                    "Mean": np.mean(finite),
                })
        pd.DataFrame(cash_rows).to_excel(writer, sheet_name="Monthly_Cash", index=False)
        
        # ================================================================
        # 6. MONTHLY_EBITDA - Full EBITDA time series
        # ================================================================
        ebitda_rows = []
        for m in range(horizon):
            col = r["ebitda_ts"][:, m]
            finite = col[np.isfinite(col)]
            if len(finite) == 0:
                ebitda_rows.append({"Month": m+1, "P0": np.nan, "P5": np.nan, "P10": np.nan,
                                    "P25": np.nan, "P50": np.nan, "P75": np.nan,
                                    "P90": np.nan, "P95": np.nan, "P100": np.nan, "Mean": np.nan})
            else:
                ebitda_rows.append({
                    "Month": m + 1,
                    "P0": np.min(finite),
                    "P5": np.percentile(finite, 5),
                    "P10": np.percentile(finite, 10),
                    "P25": np.percentile(finite, 25),
                    "P50": np.percentile(finite, 50),
                    "P75": np.percentile(finite, 75),
                    "P90": np.percentile(finite, 90),
                    "P95": np.percentile(finite, 95),
                    "P100": np.max(finite),
                    "Mean": np.mean(finite),
                })
        pd.DataFrame(ebitda_rows).to_excel(writer, sheet_name="Monthly_EBITDA", index=False)
        
        # ================================================================
        # 7. MONTHLY_DEBT - Loan balances by month
        # ================================================================
        debt_rows = []
        for m in range(horizon):
            loan1 = r["loan1_bal_ts"][:, m]
            loan2 = r["loan2_bal_ts"][:, m]
            loc = r["loc_bal_ts"][:, m]
            
            debt_rows.append({
                "Month": m + 1,
                "Loan1_P50": safe_median(loan1),
                "Loan1_P5": safe_percentile(loan1, 5),
                "Loan1_P95": safe_percentile(loan1, 95),
                "Loan2_P50": safe_median(loan2),
                "Loan2_P5": safe_percentile(loan2, 5),
                "Loan2_P95": safe_percentile(loan2, 95),
                "LOC_P50": safe_median(loc),
                "LOC_P5": safe_percentile(loc, 5),
                "LOC_P95": safe_percentile(loc, 95),
                "Total_Borrower_Debt_P50": safe_median(loan1) + safe_median(loc),
                "Total_All_Debt_P50": safe_median(loan1) + safe_median(loan2) + safe_median(loc),
            })
        pd.DataFrame(debt_rows).to_excel(writer, sheet_name="Monthly_Debt", index=False)
        
        # ================================================================
        # 8. MONTHLY_COVENANTS - Covenant values at test months
        # ================================================================
        cov_rows = []
        for m in range(horizon):
            month_num = m + 1
            is_qe = (month_num % 3 == 0)
            is_ae = (month_num % 12 == 0)
            
            row = {
                "Month": month_num,
                "Is_Quarter_End": is_qe,
                "Is_Annual_End": is_ae,
            }
            
            # DSCR (QE)
            if is_qe:
                dscr = r["dscr_val_ts"][:, m]
                row["DSCR_P5"] = safe_percentile(dscr, 5)
                row["DSCR_P50"] = safe_median(dscr)
                row["DSCR_P95"] = safe_percentile(dscr, 95)
                
                dte = r["dte_val_ts"][:, m]
                row["DTE_P5"] = safe_percentile(dte, 5)
                row["DTE_P50"] = safe_median(dte)
                row["DTE_P95"] = safe_percentile(dte, 95)
            else:
                row["DSCR_P5"] = np.nan
                row["DSCR_P50"] = np.nan
                row["DSCR_P95"] = np.nan
                row["DTE_P5"] = np.nan
                row["DTE_P50"] = np.nan
                row["DTE_P95"] = np.nan
            
            # GDSCR (AE)
            if is_ae and month_num >= 12:
                gdscr = r["gdscr_val_ts"][:, m]
                row["GDSCR_P5"] = safe_percentile(gdscr, 5)
                row["GDSCR_P50"] = safe_median(gdscr)
                row["GDSCR_P95"] = safe_percentile(gdscr, 95)
            else:
                row["GDSCR_P5"] = np.nan
                row["GDSCR_P50"] = np.nan
                row["GDSCR_P95"] = np.nan
            
            cov_rows.append(row)
        
        pd.DataFrame(cov_rows).to_excel(writer, sheet_name="Monthly_Covenants", index=False)
        
        # ================================================================
        # 9. CURVE ANALYSIS - Portfolio collection curve and simulated outcomes
        # ================================================================
        curve_rows = []
        base_weights = r.get("base_curve_weights", np.zeros(horizon))
        expected_col = r.get("expected_collections", np.zeros(horizon))
        base_total = r.get("base_total_collections", 0.0)
        col_sigma = r.get("collections_sigma", 0.0)
        curve_h = r.get("curve_horizon", horizon)
        
        for m in range(horizon):
            col = r["portcol_ts"][:, m]
            finite = col[np.isfinite(col)]
            
            row = {
                "Month": m + 1,
                "Base_Weight": base_weights[m] if m < len(base_weights) else 0.0,
                "Expected_Collections": expected_col[m] if m < len(expected_col) else 0.0,
                "Simulated_P5": np.percentile(finite, 5) if len(finite) > 0 else np.nan,
                "Simulated_P10": np.percentile(finite, 10) if len(finite) > 0 else np.nan,
                "Simulated_P25": np.percentile(finite, 25) if len(finite) > 0 else np.nan,
                "Simulated_P50": np.percentile(finite, 50) if len(finite) > 0 else np.nan,
                "Simulated_P75": np.percentile(finite, 75) if len(finite) > 0 else np.nan,
                "Simulated_P90": np.percentile(finite, 90) if len(finite) > 0 else np.nan,
                "Simulated_P95": np.percentile(finite, 95) if len(finite) > 0 else np.nan,
                "Simulated_Mean": np.mean(finite) if len(finite) > 0 else np.nan,
            }
            curve_rows.append(row)
        
        curve_df = pd.DataFrame(curve_rows)
        curve_df.to_excel(writer, sheet_name="Curve_Analysis", index=False)
        
        # ================================================================
        # 9b. NEW CLIENT REVENUE OVERLAY
        # ================================================================
        if r.get("new_client_enabled", False):
            nc_base = r.get("new_client_base_m", np.zeros(horizon))
            nc_rows = []
            for m in range(horizon):
                nc_rows.append({
                    "Month": m + 1,
                    "New_Client_Base": nc_base[m],
                    "Cumulative": np.sum(nc_base[:m+1]),
                    "TTM": np.sum(nc_base[max(0,m-11):m+1]),
                })
            pd.DataFrame(nc_rows).to_excel(writer, sheet_name="New_Client_Overlay", index=False)
        
        # ================================================================
        # 10. CURE ANALYSIS SHEETS
        # ================================================================
        if include_cure_analysis:
            cure_results = required_changes_multi_target(r, cure_targets)
            
            # Combine GDSCR revenue and expenses into one sheet
            gdscr_rev = cure_results.get("GDSCR_revenue", pd.DataFrame())
            gdscr_exp = cure_results.get("GDSCR_expenses", pd.DataFrame())
            if not gdscr_rev.empty and not gdscr_exp.empty:
                gdscr_combined = gdscr_rev.copy()
                for col in gdscr_exp.columns:
                    if col != "Month":
                        gdscr_combined[f"Exp_{col}"] = gdscr_exp[col]
                # Rename revenue columns
                rename_map = {c: f"Rev_{c}" for c in gdscr_rev.columns if c != "Month"}
                gdscr_combined.rename(columns=rename_map, inplace=True)
                gdscr_combined.to_excel(writer, sheet_name="Cure_GDSCR", index=False)
            
            # Combine DTE revenue and expenses into one sheet
            dte_rev = cure_results.get("DTE_revenue", pd.DataFrame())
            dte_exp = cure_results.get("DTE_expenses", pd.DataFrame())
            if not dte_rev.empty and not dte_exp.empty:
                dte_combined = dte_rev.copy()
                for col in dte_exp.columns:
                    if col != "Month":
                        dte_combined[f"Exp_{col}"] = dte_exp[col]
                rename_map = {c: f"Rev_{c}" for c in dte_rev.columns if c != "Month"}
                dte_combined.rename(columns=rename_map, inplace=True)
                dte_combined.to_excel(writer, sheet_name="Cure_DTE", index=False)
        
        # ================================================================
        # 10. ASSUMPTIONS - Input echo
        # ================================================================
        assumptions_df = pd.DataFrame([
            {"Parameter": k, "Value": v} for k, v in sorted(A.items())
        ])
        assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)
        
        # ================================================================
        # 11. RUN_INFO - Metadata
        # ================================================================
        run_info = pd.DataFrame([
            {"Field": "Run Timestamp", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Field": "Debt Mode", "Value": r["debt_mode"]},
            {"Field": "Simulations", "Value": str(r["n_sims"])},
            {"Field": "Horizon Months", "Value": str(r["horizon_months"])},
            {"Field": "Checkpoints", "Value": ",".join(map(str, r["checkpoint_months"]))},
            {"Field": "Seed", "Value": str(r.get("seed", "None"))},
        ])
        run_info.to_excel(writer, sheet_name="Run_Info", index=False)
    
    print(f"\n{'='*60}")
    print(f"DATA EXPORT COMPLETE: {filepath}")
    print(f"{'='*60}")
    print("Sheets included:")
    print("  - Executive_Data      : Key metrics for executive summary links")
    print("  - Breach_Cumulative   : Cumulative breach probabilities by checkpoint")
    print("  - Breach_PointInTime  : Point-in-time breach probabilities")
    print("  - Snapshots           : P5/P25/P50/P75/P95 distributions at checkpoints")
    print("  - Monthly_Cash        : Full cash balance time series")
    print("  - Monthly_EBITDA      : Full EBITDA time series")
    print("  - Monthly_Debt        : Loan balances by month")
    print("  - Monthly_Covenants   : DSCR, DTE, GDSCR values at test months")
    print("  - Curve_Analysis      : Portfolio collection curve and simulated outcomes")
    print("  - Cure_GDSCR          : Required changes for Global DSCR cure")
    print("  - Cure_DTE            : Required changes for D/EBITDA cure")
    print("  - Assumptions         : Input assumptions echo")
    print("  - Run_Info            : Run metadata and timestamp")
    print(f"{'='*60}\n")
    
    return filepath

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Monte Carlo covenant risk model")
    p.add_argument("--assumptions", default="assumptions.csv", help="Path to assumptions.csv")
    p.add_argument("--curve", default="curve.csv", help="Path to curve.csv")
    p.add_argument("--revexp", default="RevAndExp.csv", help="Path to RevAndExp.csv (GAAP basis only)")
    p.add_argument("--seed", default=None, help="Random seed (int). If omitted, uses nondeterministic seed.")
    p.add_argument("--checkpoints", default="12,18,24,36,48,60", help="Comma-separated checkpoint months")
    p.add_argument("--output", default="mc_output_data.xlsx", help="Path for Excel data output (default: mc_output_data.xlsx)")
    p.add_argument("--no-export", action="store_true", help="Skip Excel export")
    args = p.parse_args()

    checkpoints = [int(x.strip()) for x in str(args.checkpoints).split(",") if x.strip()]
    seed = None if args.seed in (None, "", "None", "none") else int(args.seed)

    # Read once for close-check printing and defaults
    A = read_assumptions(args.assumptions)

    r = run_sim(
        curve_path=args.curve,
        assumptions_path=args.assumptions,
        revexp_path=args.revexp,
        seed=seed,
        checkpoint_months=checkpoints,
        verbose_close_check=True,
    )
    
    # Store seed in results for Run_Info sheet
    r["seed"] = seed

    # Standard reports
    print_ever_breach_summary(r)
    checkpoint_breach_table(r)
    checkpoint_snapshots_table(r)
    
    # Monthly balance summary
    print_monthly_balance_summary(r, months=[12, 24, 36, 48, 60])
    
    # Required revenue/expense changes for cure (P15, P10, P5, P0)
    print_required_changes_summary(r, targets=[0.15, 0.10, 0.05, 0.00])
    
    # Excel export - ALWAYS run unless --no-export flag
    if not args.no_export:
        export_results_to_excel(r, filepath=args.output)


# ============================================================
# Jupyter Notebook Helper Functions
# ============================================================
def run_full_analysis(assumptions_path: str = "assumptions.csv",
                      curve_path: str = "curve.csv",
                      revexp_path: str = "RevAndExp.csv",
                      seed: Optional[int] = 42,
                      checkpoints: List[int] = [12, 18, 24, 36, 48, 60],
                      excel_output: Optional[str] = "mc_output_data.xlsx",
                      cure_targets: List[float] = [0.15, 0.10, 0.05, 0.00],
                      verbose: bool = True) -> Dict:
    """
    Convenience function for Jupyter Notebook usage.
    Runs full simulation and analysis, exports to Excel, returns results dict.
    
    Usage in Jupyter:
        from mc_covenants_v12_with_export import run_full_analysis
        r = run_full_analysis(excel_output="mc_output_data.xlsx")
        
    The output file mc_output_data.xlsx can be linked to from presentation workbooks.
    """
    r = run_sim(
        curve_path=curve_path,
        assumptions_path=assumptions_path,
        revexp_path=revexp_path,
        seed=seed,
        checkpoint_months=checkpoints,
        verbose_close_check=verbose,
    )
    
    # Store seed in results for Run_Info sheet
    r["seed"] = seed
    
    if verbose:
        print_ever_breach_summary(r)
        checkpoint_breach_table(r)
        checkpoint_snapshots_table(r)
        print_monthly_balance_summary(r, months=checkpoints)
        print_required_changes_summary(r, targets=cure_targets)
    
    if excel_output:
        export_results_to_excel(r, filepath=excel_output, cure_targets=cure_targets)
    
    return r


if __name__ == "__main__":
    main()