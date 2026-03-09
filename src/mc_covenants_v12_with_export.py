
import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')


# ------------------------------------------------------------------
# Assumptions loader
# ------------------------------------------------------------------
def read_assumptions(path: str = "assumptions.csv") -> Dict[str, str]:
    df = pd.read_csv(path).dropna(subset=["name"])
    df["name"] = df["name"].astype(str).str.strip()
    df = df[~df["name"].str.startswith("#")]
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        k = str(row["name"]).strip()
        if not k or k.lower() == "nan":
            continue
        out[k] = str(row.get("value", "")).split("#", 1)[0].strip()
    return out


# ------------------------------------------------------------------
# GAAP revenue/expense loaders
# ------------------------------------------------------------------
def _to_numeric_series(s: pd.Series) -> np.ndarray:
    """Parse a column that might have commas, $, or parenthetical negatives."""
    ss = s.astype(str).str.strip()
    ss = ss.str.replace(",", "", regex=False).str.replace("$", "", regex=False)
    ss = ss.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    return pd.to_numeric(ss, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def load_revexp_series(revexp_path, horizon, start_month=None):
    """Load GAAP rev/exp from CSV. Extends beyond history by repeating last 12 months."""
    df = pd.read_csv(revexp_path)
    for c in ("Month", "Revenue", "Expenses"):
        if c not in df.columns:
            raise ValueError(f"Rev/Exp CSV missing column: {c}")
    df = df.copy()
    df["Month_dt"] = pd.to_datetime(df["Month"], format="%b-%y", errors="coerce")
    if df["Month_dt"].isna().any():
        raise ValueError(f"Unparseable months: {df.loc[df['Month_dt'].isna(), 'Month'].head(5).tolist()}")
    df = df.sort_values("Month_dt").reset_index(drop=True)
    if start_month:
        df = df.loc[df["Month_dt"] >= pd.to_datetime(start_month, format="%b-%y")].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"start_month={start_month} is after last month in CSV")

    rev = _to_numeric_series(df["Revenue"])
    exp = _to_numeric_series(df["Expenses"])
    adj = _to_numeric_series(df["one_time_adjustment"]) if "one_time_adjustment" in df.columns else np.zeros(len(df))
    if horizon <= len(rev):
        return rev[:horizon], exp[:horizon], adj[:horizon]

    rev_out, exp_out, adj_out = np.zeros(horizon), np.zeros(horizon), np.zeros(horizon)
    n = len(rev)
    rev_out[:n], exp_out[:n], adj_out[:n] = rev, exp, adj
    pat = min(12, n)
    for t in range(n, horizon):
        k = (t - n) % pat
        rev_out[t], exp_out[t] = rev[-pat + k], exp[-pat + k]
    return rev_out, exp_out, adj_out


def _compute_recency_weights(n_months, half_life_months=12):
    decay = 0.5 ** (1 / half_life_months)
    w = decay ** np.arange(n_months)[::-1]
    return w / w.sum()


def _weighted_seasonal_factors(values, cal_months, half_life_months=12):
    """12-element seasonal index, recency-weighted."""
    weights = _compute_recency_weights(len(values), half_life_months)
    wmean = np.sum(values * weights)
    seasonal = np.ones(12)
    for m in range(1, 13):
        mask = cal_months == m
        if mask.sum() > 0:
            mw = weights[mask]
            mw = mw / mw.sum()
            seasonal[m - 1] = np.sum(values[mask] * mw) / wmean if wmean > 0 else 1.0
    return seasonal


def load_revexp_series_with_forecast(
        revexp_path, horizon, start_month=None, expense_floor=425000,
        expense_floor_hold_months=9, expense_growth_annual=0.03, half_life_months=12):
    """
    Regression-based forecast: weighted seasonal decomposition + trend.
    Revenue gets a linear trend with seasonality overlay.
    Expenses decay toward a floor, hold there, then grow at the annual rate.
    Full history used for fitting; output window starts at start_month.
    """
    df = pd.read_csv(revexp_path).copy()
    for c in ("Month", "Revenue", "Expenses"):
        if c not in df.columns:
            raise ValueError(f"Rev/Exp CSV missing column: {c}")
    df["Month_dt"] = pd.to_datetime(df["Month"], format="%b-%y", errors="coerce")
    df = df.sort_values("Month_dt").reset_index(drop=True)

    rev_full = _to_numeric_series(df["Revenue"])
    exp_full = _to_numeric_series(df["Expenses"])
    adj_full = _to_numeric_series(df["one_time_adjustment"]) if "one_time_adjustment" in df.columns else np.zeros(len(df))

    n_hist = len(rev_full)
    month_idx = np.arange(n_hist)
    cal_months = df["Month_dt"].dt.month.values
    rev_seasonal = _weighted_seasonal_factors(rev_full, cal_months, half_life_months)
    exp_seasonal = _weighted_seasonal_factors(exp_full, cal_months, half_life_months)

    # Deseasonalize
    rev_ds = rev_full / np.where(rev_seasonal[cal_months - 1] > 0, rev_seasonal[cal_months - 1], 1)
    exp_ds = exp_full / np.where(exp_seasonal[cal_months - 1] > 0, exp_seasonal[cal_months - 1], 1)

    # Revenue: weighted linear regression on deseasonalized series
    weights = _compute_recency_weights(n_hist, half_life_months)
    xm = np.sum(month_idx * weights)
    ym = np.sum(rev_ds * weights)
    xy_cov = np.sum(weights * (month_idx - xm) * (rev_ds - ym))
    x_var = np.sum(weights * (month_idx - xm) ** 2)
    rev_slope = xy_cov / x_var if x_var > 0 else 0
    rev_intercept = ym - rev_slope * xm

    # Expense decay rate from recent log(level - floor) regression
    lb = min(12, n_hist)
    recent_exp, recent_idx = exp_ds[-lb:], month_idx[-lb:]
    exp_current = np.sum(recent_exp * _compute_recency_weights(lb, half_life_months=6))
    above = np.maximum(recent_exp - expense_floor, 1)
    log_above = np.log(above)
    rxm, rym = recent_idx.mean(), log_above.mean()
    d = np.sum((recent_idx - rxm) ** 2)
    slope = np.sum((recent_idx - rxm) * (log_above - rym)) / d if d > 1e-9 else 0
    exp_decay = np.clip(np.exp(slope), 0.90, 1.02)

    start_idx = 0
    if start_month:
        mask = df["Month_dt"] >= pd.to_datetime(start_month, format="%b-%y")
        start_idx = df[mask].index[0] if mask.any() else n_hist

    last_dt = df["Month_dt"].iloc[-1]
    eg_mo = (1 + expense_growth_annual) ** (1 / 12)

    rev_out, exp_out, adj_out = np.zeros(horizon), np.zeros(horizon), np.zeros(horizon)
    for t in range(horizon):
        gi = start_idx + t
        if gi < n_hist:
            rev_out[t], exp_out[t], adj_out[t] = rev_full[gi], exp_full[gi], adj_full[gi]
        else:
            mf = gi - n_hist + 1
            cm = (last_dt + pd.DateOffset(months=mf)).month
            rev_out[t] = max((rev_intercept + rev_slope * gi) * rev_seasonal[cm - 1], 0)
            if mf <= expense_floor_hold_months:
                et = expense_floor + max(0, (exp_current - expense_floor) * (exp_decay ** mf))
                et = max(et, expense_floor)
            else:
                et = expense_floor * (eg_mo ** (mf - expense_floor_hold_months))
            exp_out[t] = max(et * exp_seasonal[cm - 1], expense_floor * 0.95)
    return rev_out, exp_out, adj_out


def load_new_client_revenue(csv_path, horizon, gaap_start_month=None, realization_pct=1.0):
    """Load new-client net revenue overlay. Flat-lines after CSV ends."""
    df = pd.read_csv(csv_path)
    if "Month" not in df.columns or "total_net" not in df.columns:
        raise ValueError(f"Need 'Month' and 'total_net' columns. Got: {list(df.columns)}")
    df = df.copy()
    df["Month_dt"] = pd.to_datetime(df["Month"], format="%b-%y", errors="coerce")
    df = df.sort_values("Month_dt").reset_index(drop=True)
    net = pd.to_numeric(df["total_net"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    gsd = pd.to_datetime(gaap_start_month, format="%b-%y") if gaap_start_month else df["Month_dt"].iloc[0]
    csd = df["Month_dt"].iloc[0]
    offset = (csd.year - gsd.year) * 12 + (csd.month - gsd.month)

    out = np.zeros(horizon)
    for j in range(len(net)):
        idx = offset + j
        if 0 <= idx < horizon:
            out[idx] = net[j]
    last_idx = offset + len(net) - 1
    if last_idx < horizon - 1 and len(net) > 0:
        out[last_idx + 1:] = net[-1]
    return out * float(realization_pct)


# ------------------------------------------------------------------
# Assumption accessors
# ------------------------------------------------------------------
def f(A, key, default=0.0):
    try: return float(A.get(key, default))
    except: return float(default)

def i(A, key, default=0):
    try: return int(float(A.get(key, default)))
    except: return int(default)

def s(A, key, default=""):
    return str(A.get(key, default)).strip()


# ------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------
def fmt_money(x):
    try: return "NA" if not np.isfinite(x) else f"${float(x):,.0f}"
    except: return "NA"

def fmt_num(x, digits=2):
    try: return "NA" if not np.isfinite(x) else f"{float(x):.{digits}f}"
    except: return "NA"

def fmt_pct(x):
    try: return "NA" if not np.isfinite(x) else f"{100*float(x):.1f}%"
    except: return "NA"


# ------------------------------------------------------------------
# Safe stats
# ------------------------------------------------------------------
def _finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]

def safe_percentile(a, p, default=np.nan):
    x = _finite(a)
    return float(np.percentile(x, p)) if x.size else float(default)

def safe_median(a, default=np.nan):
    x = _finite(a)
    return float(np.median(x)) if x.size else float(default)

def safe_prob(a, predicate, default=np.nan):
    x = _finite(a)
    return float(np.mean(predicate(x))) if x.size else float(default)


# ------------------------------------------------------------------
# Stochastic helpers
# ------------------------------------------------------------------
def jitter_curve(weights, concentration=200.0, rng=None):
    """Dirichlet jitter around a probability vector."""
    rng = rng or np.random.default_rng()
    w = np.clip(np.asarray(weights, dtype=float), 1e-12, None)
    w /= w.sum()
    return rng.dirichlet(w * max(float(concentration), 1e-6))

def draw_lognormal_series(mean_arr, cv, rng=None):
    rng = rng or np.random.default_rng()
    mean_arr = np.asarray(mean_arr, dtype=float)
    if cv <= 1e-12:
        return mean_arr.copy()
    s2 = np.log1p(cv * cv)
    mu = np.log(np.clip(mean_arr, 1e-12, None)) - 0.5 * s2
    return rng.lognormal(mean=mu, sigma=np.sqrt(s2), size=mean_arr.shape)

def draw_multiplicative_lognormal(base_arr, cv, rng=None):
    """Mean-1 multiplicative lognormal noise."""
    rng = rng or np.random.default_rng()
    base_arr = np.asarray(base_arr, dtype=float)
    if cv <= 1e-12:
        return base_arr.copy()
    s2 = np.log1p(cv * cv)
    return base_arr * rng.lognormal(mean=-0.5 * s2, sigma=np.sqrt(s2), size=base_arr.shape)

def lognormal_from_mean_cv(mean, cv, rng=None):
    rng = rng or np.random.default_rng()
    if cv <= 1e-12:
        return float(mean)
    s2 = np.log1p(cv * cv)
    return float(rng.lognormal(mean=float(np.log(max(mean, 1e-12)) - 0.5 * s2), sigma=float(np.sqrt(s2))))


# ------------------------------------------------------------------
# Debt math
# ------------------------------------------------------------------
def amort_payment(principal, apr, n_months):
    r = apr / 12.0
    if n_months <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return principal / n_months
    return principal * (r * (1 + r) ** n_months) / ((1 + r) ** n_months - 1)

def scheduled_principal_next12(balance, apr, amort_months, io_months, m):
    """Forward-looking 12-month scheduled principal (current maturities proxy)."""
    bal = float(max(balance, 0.0))
    if bal <= 1e-9:
        return 0.0
    months_done = m + 1
    rem_io = max(0, int(io_months) - months_done)
    rem_amort = max(0, int(amort_months) - max(0, months_done - int(io_months)))
    r = float(apr) / 12.0
    ps = 0.0
    for _ in range(12):
        if bal <= 1e-9:
            break
        if rem_io > 0:
            rem_io -= 1
            continue
        if rem_amort <= 0:
            break
        pmt = amort_payment(bal, apr, rem_amort)
        p = max(0.0, min(bal, pmt - bal * r))
        ps += p
        bal -= p
        rem_amort -= 1
    return float(ps)


# ------------------------------------------------------------------
# Covenant calculations
# ------------------------------------------------------------------
def borrower_dscr_value(ebitda_m, borrower_interest_m, curr_maturities, m, A):
    """
    Borrower DSCR (QE test): adjusted EBITDA / (CM + interest + external DS).
    Annualized YTD until 12 months exist, then TTM.
    """
    adj_per_mo = (f(A, "cash_taxes_annual", 0) + f(A, "dividends_annual", 0) + f(A, "distributions_annual", 0)) / 12.0
    adj = ebitda_m - adj_per_mo
    w = min(m + 1, 12)
    st = m - w + 1
    num = float(np.sum(adj[st:m + 1]))
    isum = float(np.sum(borrower_interest_m[st:m + 1]))
    if w < 12:
        sc = 12.0 / w
        num *= sc
        isum *= sc
    den = isum + float(curr_maturities) + f(A, "ext_debt_annual_ds", 0.0)
    return num / den if (np.isfinite(num) and den > 1e-9) else np.nan

def debt_to_ebitda_value(ebitda_m, funded_debt_bal_m, m):
    w = min(m + 1, 12)
    st = m - w + 1
    e = float(np.sum(ebitda_m[st:m + 1]))
    if w < 12:
        e *= 12.0 / w
    return funded_debt_bal_m[m] / e if (np.isfinite(e) and e > 1e-9) else np.nan

def global_dscr_value(ebitda_m, borrower_interest_m, curr_maturities, personal_ds_m, m, A):
    """
    Global DSCR (annual test, TTM).
    Numerator adds personal cash flow; denominator adds personal P&I.
    """
    if m < 11:
        return np.nan
    adj_per_mo = (f(A, "cash_taxes_annual", 0) + f(A, "dividends_annual", 0) + f(A, "distributions_annual", 0)) / 12.0
    adj = ebitda_m - adj_per_mo
    st = m - 11
    num = float(np.sum(adj[st:m + 1])) + f(A, "personal_cash_flow_annual", 0.0)
    den = (float(np.sum(borrower_interest_m[st:m + 1])) + float(curr_maturities)
           + float(np.sum(personal_ds_m[st:m + 1])) + f(A, "extra_personal_ds_annual", 0.0))
    return num / den if (np.isfinite(num) and den > 1e-9) else np.nan

def current_ratio_value(cash, restricted_cash_bal, incl_restricted, port_col_m, m,
                        other_ca, other_cl, port_months, port_haircut, incl_rloc, rloc_bal, cm):
    ca = float(cash) + float(other_ca)
    if incl_restricted:
        ca += float(restricted_cash_bal)
    end = min(len(port_col_m), m + 1 + max(0, int(port_months)))
    if m + 1 < end:
        ca += float(port_haircut) * float(np.sum(port_col_m[m + 1:end]))
    cl = float(other_cl) + float(cm)
    if incl_rloc:
        cl += float(rloc_bal)
    return float("inf") if cl <= 1e-9 else ca / cl


# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
def run_sim(curve_path="curve.csv", assumptions_path="assumptions.csv",
            revexp_path="RevAndExp.csv", seed=None,
            checkpoint_months=None, verbose_close_check=True):

    if seed is not None:
        np.random.seed(int(seed))

    A = read_assumptions(assumptions_path)
    horizon = i(A, "horizon_months", 36)
    n_sims = i(A, "n_sims", 10000)
    checkpoint_months = checkpoint_months or [12, 18, 24, 36]
    checkpoint_months = [m for m in checkpoint_months if 1 <= m <= horizon]

    # -- Curve --
    curve = pd.read_csv(curve_path)
    if "weight" not in curve.columns:
        raise ValueError("curve.csv needs 'weight' column")
    bw_all = np.clip(curve["weight"].astype(float).to_numpy(), 0.0, None)
    curve_h = max(1, min(i(A, "portfolio_curve_horizon_months", min(36, len(bw_all))), len(bw_all), horizon))
    base_w = bw_all[:curve_h]
    if base_w.sum() <= 0:
        raise ValueError("Curve weights sum to 0")
    base_w /= base_w.sum()
    curve_conc = f(A, "curve_jitter_concentration", 200.0)

    # -- Operating baseline --
    acct = str(A.get("accounting_basis", "cash")).strip().lower().strip(",")
    gaap_start = A.get("gaap_start_month", "").strip() or None
    gaap_adj_is_cash = (i(A, "gaap_adjustment_is_cash", 0) == 1)
    rev_cv = f(A, "revenue_sigma", 0.0)
    exp_cv = f(A, "expenses_sigma", 0.0)
    growth_m = (1.0 + f(A, "business_growth_annual", 0.0)) ** (1 / 12) - 1.0
    exp_growth_ann = f(A, "expense_growth_annual", 0.03)
    midx = np.arange(horizon, dtype=float)
    gaap_adj_m = np.zeros(horizon)

    if acct == "gaap":
        rp = revexp_path
        if not os.path.isabs(rp):
            rp = os.path.join(os.path.dirname(os.path.abspath(assumptions_path)), rp)
        fm = s(A, "forecast_mode", "static").lower()
        efloor = f(A, "expense_floor", 425000)
        ehold = i(A, "expense_floor_hold_months", 9)
        fhl = i(A, "forecast_half_life_months", 12)
        if fm == "regression":
            rb, eb, ga = load_revexp_series_with_forecast(
                rp, horizon, gaap_start, efloor, ehold, exp_growth_ann, fhl)
            rev_mean, exp_mean = rb, eb
        else:
            rb, eb, ga = load_revexp_series(rp, horizon, gaap_start)
            rev_mean = rb * ((1 + growth_m) ** midx)
            egm = (1.0 + exp_growth_ann) ** (1 / 12) - 1.0
            exp_mean = eb * ((1 + egm) ** midx)
        gaap_adj_m = ga
    else:
        rev_mean = f(A, "base_cash_revenue_monthly", 0.0) * ((1 + growth_m) ** midx)
        exp_mean = f(A, "base_cash_expenses_monthly", 0.0) * ((1 + growth_m) ** midx)
        if abs(exp_growth_ann) > 1e-12:
            egm = (1.0 + exp_growth_ann) ** (1 / 12) - 1.0
            exp_mean *= ((1 + egm) ** midx)

    # -- New client overlay --
    nc_on = (i(A, "new_client_enabled", 0) == 1)
    nc_base = np.zeros(horizon)
    nc_cv = 0.0
    if nc_on:
        ncp = s(A, "new_client_csv", "new_client_revenue.csv")
        if not os.path.isabs(ncp):
            ncp = os.path.join(os.path.dirname(os.path.abspath(assumptions_path)), ncp)
        nc_real = f(A, "new_client_realization_pct", 1.0)
        nc_cv = f(A, "new_client_sigma", 0.0)
        nc_base = load_new_client_revenue(ncp, horizon, gaap_start, nc_real)
        if verbose_close_check:
            print(f"New client overlay: ON (real={nc_real:.0%}, CV={nc_cv:.2f}, total={fmt_money(np.sum(nc_base))})")

    # -- Cash basis split --
    # When on, covenant EBITDA uses GAAP while the cash waterfall uses separate
    # CF-derived baselines. Keeps accrual timing from distorting the cash position.
    cb_split = (i(A, "cash_basis_split", 0) == 1)
    cb_rev_base = np.zeros(horizon)
    cb_exp_base = np.zeros(horizon)
    cb_rcv = f(A, "cash_basis_rev_sigma", 0.0)
    cb_ecv = f(A, "cash_basis_exp_sigma", 0.0)
    if cb_split:
        cbp = s(A, "cash_basis_csv", "cash_basis_baseline.csv")
        if not os.path.isabs(cbp):
            cbp = os.path.join(os.path.dirname(os.path.abspath(assumptions_path)), cbp)
        cbd = pd.read_csv(cbp).sort_values("month").reset_index(drop=True)
        n_cb = len(cbd)
        cr_arr = cbd["cash_revenue"].astype(float).to_numpy()
        ce_arr = cbd["cash_expenses"].astype(float).to_numpy()
        if n_cb >= horizon:
            cb_rev_base[:] = cr_arr[:horizon]
            cb_exp_base[:] = ce_arr[:horizon]
        else:
            cb_rev_base[:n_cb] = cr_arr
            cb_exp_base[:n_cb] = ce_arr
            pat = min(12, n_cb)
            for t in range(n_cb, horizon):
                k = (t - n_cb) % pat
                cb_rev_base[t] = cr_arr[-pat + k]
                cb_exp_base[t] = ce_arr[-pat + k]
        if cb_rcv <= 1e-12:
            cb_rcv = rev_cv
        if cb_ecv <= 1e-12:
            cb_ecv = exp_cv
        if verbose_close_check:
            print(f"Cash basis split: ON  net avg M1-12={fmt_money((cb_rev_base[:12] - cb_exp_base[:12]).mean())}/mo")

    # -- Portfolio --
    base_total_col = f(A, "base_total_collections", 0.0)
    col_cv = f(A, "collections_sigma", 0.0)
    step_gain = sum(f(A, k, 0.0) for k in (
        "gain_paydown_direct_monthly", "gain_payoff_ap_monthly",
        "gain_absorb_loan_monthly", "gain_absorb_loc_monthly"))

    # -- Debt terms --
    io_mo = i(A, "term_io_months", 0)
    l1p0 = f(A, "loan1_principal", 0.0)
    l1r = f(A, "loan1_apr", 0.0)
    l1a = i(A, "loan1_amort_months", 60)
    l2p0 = f(A, "loan2_principal", 0.0)
    l2r = f(A, "loan2_apr", 0.0)
    l2a = i(A, "loan2_amort_months", 84)

    # If after-IO amort not explicitly given, derive from total term minus IO
    l1aio = i(A, "loan1_amort_months_after_io", 0) or max(1, l1a - io_mo)
    l2aio = i(A, "loan2_amort_months_after_io", 0) or max(1, l2a - io_mo)
    l1pmt = amort_payment(l1p0, l1r, l1aio) if l1aio > 0 else 0.0
    l2pmt = amort_payment(l2p0, l2r, l2aio) if l2aio > 0 else 0.0

    # Revolver
    loc_limit = f(A, "loc_limit", 0.0)
    loc_apr = f(A, "loc_apr", 0.0)
    loc_r = loc_apr / 12.0
    loc_draw0 = f(A, "loc_initial_draw", 0.0)
    loc_autopay = (i(A, "loc_auto_paydown", 1) == 1)
    loc_backstop = (i(A, "loc_draw_to_prevent_cash_negative", 0) == 1)
    loc_buffer = f(A, "loc_min_cash_buffer", 0.0)
    rest_enabled = (i(A, "rloc_rest_enabled", 0) == 1)

    # -- Sources & uses --
    one_time_uses = sum(f(A, k, 0.0) for k in (
        "purchase_price", "use_paydown_direct", "use_payoff_ap",
        "use_absorb_loan_payoff", "use_absorb_loc_payoff"))
    term_proceeds = f(A, "total_new_term_debt_proceeds", 0.0)
    reserve0 = f(A, "interest_reserve_initial", 0.0)
    starting_cash = (term_proceeds + loc_draw0) - one_time_uses - reserve0 + f(A, "starting_cash_adjustment", 0.0)
    other_disb_mo = f(A, "other_cash_disbursements_monthly", 0.0)

    # -- Covenant params --
    comp_pct = f(A, "comp_balance_pct", 0.20)
    comp_cash_ok = (i(A, "comp_balance_cash_eligible", 1) == 1)
    comp_incl_personal = (i(A, "comp_balance_include_personal", 1) == 1)
    avg_trust = f(A, "avg_trust_balance", 0.0)
    trust_hc = f(A, "trust_haircut", 1.0)
    cr_min = f(A, "current_ratio_min", 1.25)
    oca = f(A, "other_current_assets", 0.0)
    ocl = f(A, "other_current_liabilities", 0.0)
    pcm = i(A, "portfolio_current_months", 12)
    pch = f(A, "portfolio_current_haircut", 1.0)
    incl_rloc_cl = (i(A, "current_ratio_include_rloc_in_cl", 1) == 1)
    incl_rc_ca = (i(A, "current_ratio_include_restricted_cash", 1) == 1)
    rc_bal = f(A, "restricted_cash_balance", 0.0)
    dscr_min = f(A, "dscr_min", 1.30)
    gdscr_min = f(A, "gdscr_min", 2.0)
    dte_max_12 = f(A, "debt_to_ebitda_max_first12", 2.25)
    dte_max_after = f(A, "debt_to_ebitda_max_after12", 1.75)

    if verbose_close_check:
        print(f"\n---- CLOSE CHECK ----")
        print(f"Term proceeds: {fmt_money(term_proceeds)}  Uses: {fmt_money(one_time_uses)}  Starting cash: {fmt_money(starting_cash)}")
        print(f"LOC: {fmt_money(loc_draw0)} drawn / {fmt_money(loc_limit)} limit")
        if starting_cash < 0:
            print(f"WARNING: Starting cash NEGATIVE ({fmt_money(starting_cash)})")
        print(f"---------------------\n")

    # -- Allocate output arrays --
    shape = (n_sims, horizon)
    cash_ts = np.full(shape, np.nan)
    ebitda_ts = np.full(shape, np.nan)
    revenue_ts = np.full(shape, np.nan)
    opex_ts = np.full(shape, np.nan)
    portcol_ts = np.full(shape, np.nan)
    bint_ts = np.full(shape, np.nan)    # borrower interest
    pds_ts = np.full(shape, np.nan)     # personal debt service
    fdb_ts = np.full(shape, np.nan)     # funded debt balance
    cm_ts = np.full(shape, np.nan)      # current maturities (test months only)
    l1b_ts = np.full(shape, np.nan)     # loan1 balance
    l2b_ts = np.full(shape, np.nan)     # loan2 balance
    loc_ts = np.full(shape, np.nan)     # LOC balance
    cr_ts = np.full(shape, np.nan)
    dscr_ts = np.full(shape, np.nan)
    dte_ts = np.full(shape, np.nan)
    gdscr_ts = np.full(shape, np.nan)
    comp_br = np.full(shape, np.nan)
    cr_br = np.full(shape, np.nan)
    dscr_br = np.full(shape, np.nan)
    dte_br = np.full(shape, np.nan)
    gdscr_br = np.full(shape, np.nan)
    rest_br = np.full(shape, np.nan)

    # Independent RNG streams so changing horizon doesn't shift earlier draws
    master_ss = np.random.SeedSequence(int(seed) if seed is not None else None)

    for sim in range(n_sims):
        sim_ss = master_ss.spawn(1)[0]
        rng_port = np.random.default_rng(sim_ss.spawn(1)[0])
        rng_rev = np.random.default_rng(sim_ss.spawn(1)[0])
        rng_exp = np.random.default_rng(sim_ss.spawn(1)[0])
        rng_cr = np.random.default_rng(sim_ss.spawn(1)[0])
        rng_ce = np.random.default_rng(sim_ss.spawn(1)[0])

        # Portfolio collections
        total_col = lognormal_from_mean_cv(base_total_col, col_cv, rng=rng_port)
        w = jitter_curve(base_w, concentration=curve_conc, rng=rng_port)
        port_col = np.zeros(horizon)
        nw = min(curve_h, horizon)
        if nw > 0:
            port_col[:nw] = total_col * w[:nw]

        # Revenue / expenses
        if acct == "gaap":
            revenue = draw_multiplicative_lognormal(rev_mean, rev_cv, rng=rng_rev)
            opex = draw_multiplicative_lognormal(exp_mean, exp_cv, rng=rng_exp)
        else:
            revenue = draw_lognormal_series(rev_mean, rev_cv, rng=rng_rev)
            opex = draw_lognormal_series(exp_mean, exp_cv, rng=rng_exp)

        revenue_ts[sim] = revenue
        opex_ts[sim] = opex
        portcol_ts[sim] = port_col

        gaap_col_in_rev = (i(A, "gaap_revenue_includes_collections", 0) == 1)
        port_pnl = np.zeros_like(port_col) if (acct == "gaap" and gaap_col_in_rev) else port_col

        # New client stochastic overlay
        if nc_on and np.any(nc_base > 0):
            nc_sim = draw_multiplicative_lognormal(nc_base, nc_cv, rng=rng_rev)
            nc_sim = np.where(nc_base > 0, nc_sim, 0.0)
        else:
            nc_sim = np.zeros(horizon)

        # Covenant EBITDA vs cash waterfall EBITDA
        ebitda_cov = (revenue - opex + port_pnl) + step_gain + gaap_adj_m + nc_sim
        if cb_split:
            cr_sim = draw_multiplicative_lognormal(cb_rev_base, cb_rcv, rng=rng_cr)
            ce_sim = draw_multiplicative_lognormal(cb_exp_base, cb_ecv, rng=rng_ce)
            ebitda_cash = (cr_sim - ce_sim + port_col) + step_gain + nc_sim
        else:
            ebitda_cash = (revenue - opex + port_pnl) + step_gain + (gaap_adj_m if gaap_adj_is_cash else 0.0) + nc_sim

        ebitda = ebitda_cov
        ebitda_ts[sim] = ebitda_cov

        # Initialize balances
        cash = float(starting_cash)
        l1b = float(l1p0)
        l2b = float(l2p0)
        locb = float(loc_draw0)
        bint_m = np.zeros(horizon)
        fdb_m = np.zeros(horizon)
        pds_m = np.zeros(horizon)

        for m in range(horizon):
            mn = m + 1

            # Interest on beginning-of-month balances
            l1i = l1b * (l1r / 12.0)
            l2i = l2b * (l2r / 12.0)
            loci = locb * loc_r

            # Principal: IO first, then amortize
            if mn <= io_mo:
                l1p = 0.0
                l2p = 0.0
                pds_m[m] = float(l2i)
            else:
                ae = mn - io_mo - 1
                l1rem = max(0, l1aio - ae) if l1b > 1e-9 else 0
                l2rem = max(0, l2aio - ae) if l2b > 1e-9 else 0
                l1pm = l1pmt if l1rem > 0 else 0.0
                l2pm = l2pmt if l2rem > 0 else 0.0
                l1p = max(0.0, min(l1b, l1pm - l1i))
                l2p = max(0.0, min(l2b, l2pm - l2i))
                pds_m[m] = float(l2i + l2p)

            # Cash: business pays all DS
            ds = (l1i + l1p) + (l2i + l2p) + loci
            cash += float(ebitda_cash[m]) - ds - other_disb_mo

            # LOC liquidity backstop
            if loc_backstop and loc_limit > 0 and cash < loc_buffer:
                draw = min(loc_buffer - cash, max(0.0, loc_limit - locb))
                if draw > 0:
                    locb += draw
                    cash += draw

            l1b = max(0.0, l1b - l1p)
            l2b = max(0.0, l2b - l2p)

            # Auto-paydown revolver from excess cash
            if loc_autopay and locb > 0:
                cbase = (l1b + l2b + locb) if comp_incl_personal else (l1b + locb)
                floor = comp_pct * cbase
                pay = min(max(0.0, cash - floor), locb)
                if pay > 0:
                    locb -= pay
                    cash -= pay

            # Record state
            cash_ts[sim, m] = cash
            l1b_ts[sim, m] = l1b
            l2b_ts[sim, m] = l2b
            loc_ts[sim, m] = locb

            # Borrower-only interest and funded debt (loan2 excluded from these covenants)
            bint_m[m] = l1i + loci
            fdb_m[m] = l1b + locb

            # Compensating balance (monthly)
            cbase = (l1b + l2b + locb) if comp_incl_personal else (l1b + locb)
            req_dep = comp_pct * cbase
            avail = (cash if comp_cash_ok else 0.0) + avg_trust * trust_hc
            comp_br[sim, m] = 0.0 if avail >= (req_dep - 1e-6) else 1.0

            # RLOC rest: at annual boundaries, check if LOC hit zero in prior year
            if rest_enabled:
                if mn % 12 == 0:
                    ws = max(0, m - 11)
                    rested = any(loc_ts[sim, k] <= 1e-6 for k in range(ws, m + 1)) or locb <= 1e-6
                    rest_br[sim, m] = 0.0 if rested else 1.0
                else:
                    rest_br[sim, m] = 0.0
            else:
                rest_br[sim, m] = 0.0

            # Quarter-end covenant tests
            is_qe = (mn % 3 == 0)
            is_ae = (mn % 12 == 0)

            if is_qe:
                cm_b = scheduled_principal_next12(l1b, l1r, l1aio, io_mo, m)
                cm_ts[sim, m] = cm_b

                cr = current_ratio_value(
                    cash, rc_bal, incl_rc_ca, port_col, m,
                    oca, ocl, pcm, pch, incl_rloc_cl, locb, cm_b)
                cr_ts[sim, m] = cr
                cr_br[sim, m] = 1.0 if (np.isfinite(cr) and cr < cr_min) else 0.0

                # DSCR not tested during IO
                if mn > io_mo:
                    d = borrower_dscr_value(ebitda, bint_m, cm_b, m, A)
                    dscr_ts[sim, m] = d
                    dscr_br[sim, m] = 1.0 if (np.isfinite(d) and d < dscr_min) else 0.0

                # D/E tested after IO and once 12 months exist
                if mn > io_mo and mn >= 12:
                    de = debt_to_ebitda_value(ebitda, fdb_m, m)
                    dte_ts[sim, m] = de
                    dmax = dte_max_12 if mn <= 12 else dte_max_after
                    dte_br[sim, m] = 1.0 if (np.isfinite(de) and de > dmax) else 0.0

            # Annual global DSCR
            # Borrower-only CM here; loan2 P&I already in pds_m so no double-count
            if is_ae and mn >= 12:
                cm_g = scheduled_principal_next12(l1b, l1r, l1aio, io_mo, m)
                gd = global_dscr_value(ebitda, bint_m, cm_g, pds_m, m, A)
                gdscr_ts[sim, m] = gd
                gdscr_br[sim, m] = 1.0 if (np.isfinite(gd) and gd < gdscr_min) else 0.0

        bint_ts[sim] = bint_m
        pds_ts[sim] = pds_m

    # -- Aggregate --
    def ever_breach(ts):
        tested = np.isfinite(ts)
        return float(np.mean(np.any((ts > 0.5) & tested, axis=1)))

    bcf = np.zeros(horizon)
    bcf[:curve_h] = base_w

    return {
        "assumptions": A, "n_sims": n_sims, "horizon_months": horizon,
        "checkpoint_months": checkpoint_months,
        "base_curve_weights": bcf, "base_total_collections": base_total_col,
        "expected_collections": base_total_col * bcf,
        "curve_horizon": curve_h, "collections_sigma": col_cv,
        "new_client_enabled": nc_on, "new_client_base_m": nc_base,
        "cash_ts": cash_ts, "ebitda_ts": ebitda_ts,
        "revenue_ts": revenue_ts, "opex_ts": opex_ts, "portcol_ts": portcol_ts,
        "borrower_interest_ts": bint_ts, "personal_ds_ts": pds_ts,
        "funded_debt_bal_ts": fdb_ts, "curr_maturities_ts": cm_ts,
        "cr_val_ts": cr_ts, "dscr_val_ts": dscr_ts,
        "dte_val_ts": dte_ts, "gdscr_val_ts": gdscr_ts,
        "loan1_bal_ts": l1b_ts, "loan2_bal_ts": l2b_ts, "loc_bal_ts": loc_ts,
        "comp_breach_ts": comp_br, "cr_breach_ts": cr_br,
        "dscr_breach_ts": dscr_br, "dte_breach_ts": dte_br,
        "gdscr_breach_ts": gdscr_br, "rest_breach_ts": rest_br,
        "fail_comp_pct": ever_breach(comp_br),
        "fail_current_ratio_pct": ever_breach(cr_br),
        "fail_dscr_pct": ever_breach(dscr_br),
        "fail_debt_to_ebitda_pct": ever_breach(dte_br),
        "fail_gdscr_pct": ever_breach(gdscr_br),
        "fail_rloc_rest_pct": ever_breach(rest_br),
        "fail_cash_negative_ever_pct": float(np.any(cash_ts < 0, axis=1).mean()),
        "fail_ebitda_negative_ever_pct": float(np.any(ebitda_ts < 0, axis=1).mean()),
    }


# ------------------------------------------------------------------
# Console reporting
# ------------------------------------------------------------------
def print_ever_breach_summary(r):
    print("---- MONTE CARLO COVENANT SUMMARY ----")
    print(f"Sims: {r['n_sims']:,} | Horizon: {r['horizon_months']} months\n")
    print("Ever-breach probabilities:")
    for label, key in [("Comp balance", "fail_comp_pct"), ("Current ratio", "fail_current_ratio_pct"),
                       ("Borrower DSCR", "fail_dscr_pct"), ("Debt/EBITDA", "fail_debt_to_ebitda_pct"),
                       ("Global DSCR", "fail_gdscr_pct"), ("RLOC rest", "fail_rloc_rest_pct"),
                       ("Cash < 0", "fail_cash_negative_ever_pct"), ("EBITDA < 0", "fail_ebitda_negative_ever_pct")]:
        print(f"  {label:<24} {fmt_pct(r[key])}")
    print()


def checkpoint_breach_table(r):
    """Cumulative breach probability through each checkpoint."""
    def p_through(ts_name, idx):
        w = r[ts_name][:, :idx + 1]
        tested = np.isfinite(w)
        return float(np.mean(np.any((w > 0.5) & tested, axis=1))) if np.any(tested) else np.nan

    rows = []
    for mth in r["checkpoint_months"]:
        idx = mth - 1
        rows.append({"Month": mth,
                      "Comp": p_through("comp_breach_ts", idx),
                      "CR": p_through("cr_breach_ts", idx),
                      "DSCR": p_through("dscr_breach_ts", idx),
                      "D/E": p_through("dte_breach_ts", idx),
                      "GDSCR": p_through("gdscr_breach_ts", idx),
                      "Rest": p_through("rest_breach_ts", idx),
                      "Cash<0": float(np.mean(np.any(r["cash_ts"][:, :idx + 1] < 0, axis=1))),
                      "EBITDA<0": float(np.mean(np.any(r["ebitda_ts"][:, :idx + 1] < 0, axis=1)))})
    df = pd.DataFrame(rows)
    print("---- CUMULATIVE BREACH THROUGH CHECKPOINT ----")
    for _, row in df.iterrows():
        print(f"  M{int(row['Month']):>3}: Comp={fmt_pct(row['Comp'])}  CR={fmt_pct(row['CR'])}  "
              f"DSCR={fmt_pct(row['DSCR'])}  D/E={fmt_pct(row['D/E'])}  "
              f"GDSCR={fmt_pct(row['GDSCR'])}  Cash<0={fmt_pct(row['Cash<0'])}")
    print()
    return df


def checkpoint_snapshots_table(r):
    A = r["assumptions"]
    rows = []
    for mth in r["checkpoint_months"]:
        idx = mth - 1
        rows.append({
            "Month": mth,
            "Cash_P50": safe_median(r["cash_ts"][:, idx]),
            "Cash_P5": safe_percentile(r["cash_ts"][:, idx], 5),
            "DSCR_P5": safe_percentile(r["dscr_val_ts"][:, idx], 5),
            "DTE_P95": safe_percentile(r["dte_val_ts"][:, idx], 95),
            "GDSCR_P5": safe_percentile(r["gdscr_val_ts"][:, idx], 5),
        })
    df = pd.DataFrame(rows)
    print("---- CHECKPOINT SNAPSHOTS ----")
    for _, row in df.iterrows():
        print(f"  M{int(row['Month']):>3}: Cash P50={fmt_money(row['Cash_P50'])}  "
              f"DSCR P5={fmt_num(row['DSCR_P5'])}  D/E P95={fmt_num(row['DTE_P95'])}x  "
              f"GDSCR P5={fmt_num(row['GDSCR_P5'])}")
    print()
    return df


# ------------------------------------------------------------------
# Fan charts
# ------------------------------------------------------------------
def fan_bands(ts, ps=(5, 25, 50, 75, 95)):
    ts = np.asarray(ts, dtype=float)
    out = {}
    for p in ps:
        band = np.full(ts.shape[1], np.nan)
        for t in range(ts.shape[1]):
            col = _finite(ts[:, t])
            if col.size:
                band[t] = np.percentile(col, p)
        out[p] = band
    return out

def plot_fan(ts, title, ylabel):
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

def plot_all_fans(r):
    plot_fan(r["cash_ts"], "Cash (P5/P25/P50/P75/P95)", "Cash")
    plot_fan(r["ebitda_ts"], "EBITDA (P5/P25/P50/P75/P95)", "EBITDA")
    plot_fan(r["dscr_val_ts"], "DSCR (QE values)", "DSCR")
    plot_fan(r["dte_val_ts"], "Debt/EBITDA (QE values)", "Debt/EBITDA")
    plt.show()


# ------------------------------------------------------------------
# Reverse-solve: required rev uplift / expense cut for covenant cure
# ------------------------------------------------------------------
def required_change_by_checkpoint_analytic(r, covenant, mode, target_prob=0.15):
    """
    How much rev increase or expense cut brings cumulative breach <= target_prob?
    Respects actual test timing (QE for D/E, annual for GDSCR).
    """
    covenant, mode = covenant.upper(), mode.upper()
    assert covenant in {"DTE", "GDSCR"} and mode in {"REVENUE", "EXPENSES"}
    target_prob = max(float(target_prob), 1e-6)

    A = r["assumptions"]
    horizon, n_sims = int(r["horizon_months"]), int(r["n_sims"])
    ebitda_ts = np.asarray(r["ebitda_ts"], dtype=float)
    rev_ts = np.asarray(r["revenue_ts"], dtype=float)
    opex_ts = np.asarray(r["opex_ts"], dtype=float)
    bint_ts = np.asarray(r["borrower_interest_ts"], dtype=float)
    fdb_ts = np.asarray(r["funded_debt_bal_ts"], dtype=float)
    cm_ts = np.asarray(r["curr_maturities_ts"], dtype=float)
    pds_ts = np.asarray(r.get("personal_ds_ts", np.zeros_like(ebitda_ts)), dtype=float)

    gdscr_min = f(A, "gdscr_min", 2.0)
    dte_max_12 = f(A, "debt_to_ebitda_max_first12", 2.25)
    dte_max_after = f(A, "debt_to_ebitda_max_after12", 1.75)
    adj_mo = (f(A, "cash_taxes_annual", 0) + f(A, "dividends_annual", 0) + f(A, "distributions_annual", 0)) / 12.0
    pcf = f(A, "personal_cash_flow_annual", 0.0)
    epds = f(A, "extra_personal_ds_annual", 0.0)

    def ttm(mat):
        out = np.full_like(mat, np.nan, dtype=float)
        for t in range(mat.shape[1]):
            out[:, t] = np.sum(mat[:, max(0, t - 11):t + 1], axis=1)
        return out

    ebitda_ttm = ttm(ebitda_ts)
    rev_ttm, opex_ttm = ttm(rev_ts), ttm(opex_ts)
    adj_ttm = ttm(ebitda_ts - adj_mo)
    int_ttm = ttm(bint_ts)
    pds_ttm = ttm(pds_ts)

    months = np.arange(horizon, dtype=int)
    qe_mask = ((months + 1) % 3 == 0) & (months >= 11)
    ae_mask = ((months + 1) % 12 == 0) & (months >= 11)
    test_mask = qe_mask if covenant == "DTE" else ae_mask

    req = np.zeros((n_sims, horizon))
    for t in range(horizon):
        if not test_mask[t]:
            continue
        if covenant == "DTE":
            dmax = dte_max_12 if (t + 1) <= 12 else dte_max_after
            gap = np.clip(fdb_ts[:, t] / max(dmax, 1e-12) - ebitda_ttm[:, t], 0, None)
        else:
            num = adj_ttm[:, t] + pcf
            den = int_ttm[:, t] + cm_ts[:, t] + pds_ttm[:, t] + epds
            gap = np.clip(gdscr_min * den - num, 0, None)
        denom = rev_ttm[:, t] if mode == "REVENUE" else opex_ttm[:, t]
        denom = np.where(np.abs(denom) < 1e-9, np.nan, denom)
        pct = gap / denom
        req[:, t] = np.where(np.isfinite(pct), np.clip(pct, 0, None), np.inf)

    out = {}
    for chk in r["checkpoint_months"]:
        tmax = min(chk - 1, horizon - 1)
        eligible = test_mask.copy()
        eligible[tmax + 1:] = False
        if not np.any(eligible):
            out[chk] = 0.0
            continue
        per_sim = np.max(req[:, eligible], axis=1)
        val = float(np.percentile(per_sim, 100.0 * (1.0 - target_prob)))
        out[chk] = max(0.0, val) if np.isfinite(val) else 0.0
    return out


def required_changes_multi_target(r, targets=None):
    targets = targets or [0.15, 0.10, 0.05, 0.00]
    rev_mean = np.nanmean(r["revenue_ts"][:, 0])
    exp_mean = np.nanmean(r["opex_ts"][:, 0])
    results = {}
    for cov in ["DTE", "GDSCR"]:
        for md in ["REVENUE", "EXPENSES"]:
            key = f"{cov}_{md.lower()}"
            rows = []
            for mth in r["checkpoint_months"]:
                row = {"Month": mth}
                for tgt in targets:
                    t = max(tgt, 0.001)
                    try:
                        pct = required_change_by_checkpoint_analytic(r, cov, md, t).get(mth, np.nan)
                    except Exception:
                        pct = np.nan
                    lbl = f"P{int(tgt * 100)}"
                    row[f"{lbl}_pct"] = pct
                    base = rev_mean if md == "REVENUE" else exp_mean
                    row[f"{lbl}_dollars"] = base * pct if np.isfinite(pct) else np.nan
                rows.append(row)
            results[key] = pd.DataFrame(rows)
    return results


def print_required_changes_summary(r, targets=None):
    targets = targets or [0.15, 0.10, 0.05, 0.00]
    results = required_changes_multi_target(r, targets)
    print("\n" + "=" * 70)
    print("REQUIRED CHANGES FOR COVENANT CURE")
    print("=" * 70)
    for cov in ["DTE", "GDSCR"]:
        name = "Debt/EBITDA" if cov == "DTE" else "Global DSCR"
        print(f"\n---- {name} ----")
        for md in ["REVENUE", "EXPENSES"]:
            action = "Rev increase" if md == "REVENUE" else "Exp decrease"
            df = results[f"{cov}_{md.lower()}"]
            print(f"  {action}:")
            for _, row in df.iterrows():
                parts = [f"M{int(row['Month']):>3}:"]
                for tgt in targets:
                    lbl = f"P{int(tgt * 100)}"
                    pct = row.get(f"{lbl}_pct", np.nan)
                    parts.append(f"{lbl}={pct * 100:.1f}%" if np.isfinite(pct) else f"{lbl}=N/A")
                print(f"    {'  '.join(parts)}")
    print()
    return results


# ------------------------------------------------------------------
# Monthly balance summary
# ------------------------------------------------------------------
def monthly_balance_summary(r):
    horizon = r["horizon_months"]
    metrics = {"Cash": r["cash_ts"], "LOC": r["loc_bal_ts"],
               "Loan1": r["loan1_bal_ts"], "Loan2": r["loan2_bal_ts"], "EBITDA": r["ebitda_ts"]}
    pctiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    rows = []
    for m in range(horizon):
        row = {"Month": m + 1}
        for name, ts in metrics.items():
            col = _finite(ts[:, m])
            if col.size:
                for p in pctiles:
                    if p == 0: row[f"{name}_P{p}"] = np.min(col)
                    elif p == 100: row[f"{name}_P{p}"] = np.max(col)
                    else: row[f"{name}_P{p}"] = np.percentile(col, p)
            else:
                for p in pctiles:
                    row[f"{name}_P{p}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def print_monthly_balance_summary(r, months=None):
    df = monthly_balance_summary(r)
    months = months or list(range(3, r["horizon_months"] + 1, 3))
    print("\n---- BALANCE SUMMARY (P5 / P50 / P95) ----")
    for m in months:
        if m > r["horizon_months"]:
            continue
        row = df[df["Month"] == m].iloc[0]
        print(f"  M{m:>3}: Cash={fmt_money(row['Cash_P5'])}/{fmt_money(row['Cash_P50'])}/{fmt_money(row['Cash_P95'])}  "
              f"LOC={fmt_money(row['LOC_P50'])}  L1={fmt_money(row['Loan1_P50'])}  L2={fmt_money(row['Loan2_P50'])}")
    print()
    return df


# ------------------------------------------------------------------
# Excel export
# ------------------------------------------------------------------
def export_results_to_excel(r, filepath="mc_output_data.xlsx",
                            include_cure_analysis=True, cure_targets=None):
    """Single-file data export for downstream workbooks to link against."""
    from datetime import datetime
    cure_targets = cure_targets or [0.15, 0.10, 0.05, 0.00]

    A = r["assumptions"]
    horizon = r["horizon_months"]
    dscr_min = f(A, "dscr_min", 1.30)
    gdscr_min = f(A, "gdscr_min", 2.0)
    dte_max_12 = f(A, "debt_to_ebitda_max_first12", 2.25)
    dte_max_after = f(A, "debt_to_ebitda_max_after12", 1.75)

    with pd.ExcelWriter(filepath, engine='openpyxl') as w:

        # -- Executive summary --
        exec_rows = [
            {"Category": "Config", "Metric": "Simulations", "Value": str(r["n_sims"]), "Numeric": float(r["n_sims"])},
            {"Category": "Config", "Metric": "Horizon", "Value": str(horizon), "Numeric": float(horizon)},
            {"Category": "Config", "Metric": "Timestamp", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Numeric": np.nan},
        ]
        for label, key in [("Comp Balance", "fail_comp_pct"), ("Current Ratio", "fail_current_ratio_pct"),
                           ("Borrower DSCR", "fail_dscr_pct"), ("Debt/EBITDA", "fail_debt_to_ebitda_pct"),
                           ("Global DSCR", "fail_gdscr_pct"), ("RLOC Rest", "fail_rloc_rest_pct"),
                           ("Cash < 0", "fail_cash_negative_ever_pct"), ("EBITDA < 0", "fail_ebitda_negative_ever_pct")]:
            exec_rows.append({"Category": "Breach (Ever)", "Metric": label,
                              "Value": f"{r[key] * 100:.2f}%", "Numeric": r[key]})
        for label, val in [("DSCR Min", dscr_min), ("GDSCR Min", gdscr_min),
                           ("D/E Max 12mo", dte_max_12), ("D/E Max After", dte_max_after)]:
            exec_rows.append({"Category": "Thresholds", "Metric": label, "Value": str(val), "Numeric": val})
        pd.DataFrame(exec_rows).to_excel(w, sheet_name="Executive_Data", index=False)

        # -- Cumulative breach --
        def _cum_breach(ts_name, idx):
            win = r[ts_name][:, :idx + 1]
            tested = np.isfinite(win)
            return float(np.mean(np.any((win > 0.5) & tested, axis=1))) if np.any(tested) else 0.0

        breach_rows = []
        for mth in r["checkpoint_months"]:
            idx = mth - 1
            breach_rows.append({
                "Month": mth, "Comp_Balance": _cum_breach("comp_breach_ts", idx),
                "Current_Ratio": _cum_breach("cr_breach_ts", idx),
                "Borrower_DSCR": _cum_breach("dscr_breach_ts", idx),
                "Debt_to_EBITDA": _cum_breach("dte_breach_ts", idx),
                "Global_DSCR": _cum_breach("gdscr_breach_ts", idx),
                "RLOC_Rest": _cum_breach("rest_breach_ts", idx),
                "Cash_Negative": float(np.mean(np.any(r["cash_ts"][:, :idx + 1] < 0, axis=1))),
                "EBITDA_Negative": float(np.mean(np.any(r["ebitda_ts"][:, :idx + 1] < 0, axis=1)))})
        pd.DataFrame(breach_rows).to_excel(w, sheet_name="Breach_Cumulative", index=False)

        # -- Point-in-time breach --
        pit_rows = []
        for mth in r["checkpoint_months"]:
            idx = mth - 1
            def _pit(ts_name):
                v = r[ts_name][:, idx]
                tested = np.isfinite(v)
                return float(np.mean((v > 0.5) & tested)) if np.any(tested) else 0.0
            pit_rows.append({
                "Month": mth, "Comp_Balance": _pit("comp_breach_ts"),
                "Current_Ratio": _pit("cr_breach_ts"), "Borrower_DSCR": _pit("dscr_breach_ts"),
                "Debt_to_EBITDA": _pit("dte_breach_ts"), "Global_DSCR": _pit("gdscr_breach_ts"),
                "RLOC_Rest": _pit("rest_breach_ts"),
                "Cash_Negative": float(np.mean(r["cash_ts"][:, idx] < 0)),
                "EBITDA_Negative": float(np.mean(r["ebitda_ts"][:, idx] < 0))})
        pd.DataFrame(pit_rows).to_excel(w, sheet_name="Breach_PointInTime", index=False)

        # -- Snapshots --
        snap_rows = []
        for mth in r["checkpoint_months"]:
            idx = mth - 1
            row = {"Month": mth}
            for prefix, ts_name in [("Cash", "cash_ts"), ("EBITDA", "ebitda_ts"),
                                     ("DSCR", "dscr_val_ts"), ("DTE", "dte_val_ts"), ("GDSCR", "gdscr_val_ts")]:
                col = r[ts_name][:, idx]
                for p in [5, 25, 50, 75, 95]:
                    row[f"{prefix}_P{p}"] = safe_percentile(col, p)
            snap_rows.append(row)
        pd.DataFrame(snap_rows).to_excel(w, sheet_name="Snapshots", index=False)

        # -- Monthly time series helper --
        def _monthly_pctiles(ts, name):
            rows = []
            for m in range(horizon):
                col = _finite(ts[:, m])
                row = {"Month": m + 1}
                if col.size:
                    for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
                        row[f"{name}_P{p}"] = (np.min(col) if p == 0 else np.max(col) if p == 100
                                                else np.percentile(col, p))
                    row[f"{name}_Mean"] = np.mean(col)
                rows.append(row)
            return pd.DataFrame(rows)

        _monthly_pctiles(r["cash_ts"], "Cash").to_excel(w, sheet_name="Monthly_Cash", index=False)
        _monthly_pctiles(r["ebitda_ts"], "EBITDA").to_excel(w, sheet_name="Monthly_EBITDA", index=False)

        # -- Debt balances --
        debt_rows = []
        for m in range(horizon):
            debt_rows.append({
                "Month": m + 1,
                **{f"Loan1_{s}": safe_percentile(r["loan1_bal_ts"][:, m], p)
                   for s, p in [("P5", 5), ("P50", 50), ("P95", 95)]},
                **{f"Loan2_{s}": safe_percentile(r["loan2_bal_ts"][:, m], p)
                   for s, p in [("P5", 5), ("P50", 50), ("P95", 95)]},
                **{f"LOC_{s}": safe_percentile(r["loc_bal_ts"][:, m], p)
                   for s, p in [("P5", 5), ("P50", 50), ("P95", 95)]},
                "Borrower_Debt_P50": safe_median(r["loan1_bal_ts"][:, m]) + safe_median(r["loc_bal_ts"][:, m]),
                "Total_Debt_P50": (safe_median(r["loan1_bal_ts"][:, m]) + safe_median(r["loan2_bal_ts"][:, m])
                                   + safe_median(r["loc_bal_ts"][:, m])),
            })
        pd.DataFrame(debt_rows).to_excel(w, sheet_name="Monthly_Debt", index=False)

        # -- Covenant values at test months --
        cov_rows = []
        for m in range(horizon):
            mn = m + 1
            is_qe = (mn % 3 == 0)
            is_ae = (mn % 12 == 0)
            row = {"Month": mn, "Is_QE": is_qe, "Is_AE": is_ae}
            if is_qe:
                for nm, ts in [("DSCR", r["dscr_val_ts"]), ("DTE", r["dte_val_ts"])]:
                    for sn, p in [("P5", 5), ("P50", 50), ("P95", 95)]:
                        row[f"{nm}_{sn}"] = safe_percentile(ts[:, m], p)
            if is_ae and mn >= 12:
                for sn, p in [("P5", 5), ("P50", 50), ("P95", 95)]:
                    row[f"GDSCR_{sn}"] = safe_percentile(r["gdscr_val_ts"][:, m], p)
            cov_rows.append(row)
        pd.DataFrame(cov_rows).to_excel(w, sheet_name="Monthly_Covenants", index=False)

        # -- Curve analysis --
        bw = r.get("base_curve_weights", np.zeros(horizon))
        ec = r.get("expected_collections", np.zeros(horizon))
        curve_rows = []
        for m in range(horizon):
            col = _finite(r["portcol_ts"][:, m])
            row = {"Month": m + 1, "Base_Weight": bw[m], "Expected": ec[m]}
            if col.size:
                for sn, p in [("P5", 5), ("P25", 25), ("P50", 50), ("P75", 75), ("P95", 95)]:
                    row[f"Sim_{sn}"] = np.percentile(col, p)
                row["Sim_Mean"] = np.mean(col)
            curve_rows.append(row)
        pd.DataFrame(curve_rows).to_excel(w, sheet_name="Curve_Analysis", index=False)

        # -- New client overlay --
        if r.get("new_client_enabled", False):
            nc = r.get("new_client_base_m", np.zeros(horizon))
            pd.DataFrame([{"Month": m + 1, "Base": nc[m], "Cumulative": np.sum(nc[:m + 1]),
                           "TTM": np.sum(nc[max(0, m - 11):m + 1])} for m in range(horizon)]
                         ).to_excel(w, sheet_name="New_Client_Overlay", index=False)

        # -- Cure analysis --
        if include_cure_analysis:
            cure = required_changes_multi_target(r, cure_targets)
            for ck, sn in [("GDSCR", "Cure_GDSCR"), ("DTE", "Cure_DTE")]:
                rdf = cure.get(f"{ck}_revenue", pd.DataFrame())
                edf = cure.get(f"{ck}_expenses", pd.DataFrame())
                if not rdf.empty and not edf.empty:
                    combined = rdf.rename(columns={c: f"Rev_{c}" for c in rdf.columns if c != "Month"}).copy()
                    for c in edf.columns:
                        if c != "Month":
                            combined[f"Exp_{c}"] = edf[c]
                    combined.to_excel(w, sheet_name=sn, index=False)

        # -- Assumptions + run info --
        pd.DataFrame([{"Parameter": k, "Value": v} for k, v in sorted(A.items())]
                     ).to_excel(w, sheet_name="Assumptions", index=False)
        pd.DataFrame([
            {"Field": "Timestamp", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Field": "Simulations", "Value": str(r["n_sims"])},
            {"Field": "Horizon", "Value": str(horizon)},
            {"Field": "Checkpoints", "Value": ",".join(map(str, r["checkpoint_months"]))},
            {"Field": "Seed", "Value": str(r.get("seed", "None"))},
        ]).to_excel(w, sheet_name="Run_Info", index=False)

    print(f"\nExport complete: {filepath}")
    return filepath


# ------------------------------------------------------------------
# CLI + notebook entry points
# ------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Monte Carlo covenant risk model")
    p.add_argument("--assumptions", default="assumptions.csv")
    p.add_argument("--curve", default="curve.csv")
    p.add_argument("--revexp", default="RevAndExp.csv")
    p.add_argument("--seed", default=None)
    p.add_argument("--checkpoints", default="12,18,24,36,48,60")
    p.add_argument("--output", default="mc_output_data.xlsx")
    p.add_argument("--no-export", action="store_true")
    args = p.parse_args()

    checkpoints = [int(x.strip()) for x in args.checkpoints.split(",") if x.strip()]
    seed = None if args.seed in (None, "", "None", "none") else int(args.seed)

    r = run_sim(curve_path=args.curve, assumptions_path=args.assumptions,
                revexp_path=args.revexp, seed=seed, checkpoint_months=checkpoints)
    r["seed"] = seed

    print_ever_breach_summary(r)
    checkpoint_breach_table(r)
    checkpoint_snapshots_table(r)
    print_monthly_balance_summary(r, months=[12, 24, 36, 48, 60])
    print_required_changes_summary(r)

    if not args.no_export:
        export_results_to_excel(r, filepath=args.output)


def run_full_analysis(assumptions_path="assumptions.csv", curve_path="curve.csv",
                      revexp_path="RevAndExp.csv", seed=42,
                      checkpoints=None, excel_output="mc_output_data.xlsx",
                      cure_targets=None, verbose=True):
    """One-call entry point for notebooks. Runs sim, reports, exports."""
    checkpoints = checkpoints or [12, 18, 24, 36, 48, 60]
    cure_targets = cure_targets or [0.15, 0.10, 0.05, 0.00]

    r = run_sim(curve_path=curve_path, assumptions_path=assumptions_path,
                revexp_path=revexp_path, seed=seed, checkpoint_months=checkpoints,
                verbose_close_check=verbose)
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
