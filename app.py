import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import attrgetter
from pathlib import Path

st.set_page_config(page_title="Sales Analytics (Superstore)", layout="wide")
st.title("📊 Sales Analytics Dashboard — Superstore")
st.caption("Upload your Superstore CSV or place it at ./data/Superstore.csv")

# ---------- Data loader ----------
CANDIDATES = [
    Path("data/Superstore.csv"),
    Path("Superstore.csv"),
    Path("Sample-Superstore.csv"),
    Path("Sample - Superstore.csv"),
]

def find_local_csv():
    for p in CANDIDATES:
        if p.exists():
            return p
    return None

@st.cache_data
def load_csv(src):
    for enc in ("utf-8", "latin1"):
        try:
            df = pd.read_csv(src, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not read CSV with utf-8 or latin1 encoding.")

    # standardise names
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    needed = ["order_id","order_date","sales","profit","category","sub_category","customer_name","region","segment"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])
    return df

uploaded = st.file_uploader("Upload CSV", type=["csv"])
local = find_local_csv()

if uploaded is not None:
    df = load_csv(uploaded)
    st.success(f"Loaded from upload: {uploaded.name}")
elif local:
    df = load_csv(local)
    st.success(f"Loaded from file: {local}")
else:
    st.warning("No CSV found. Upload a file, or put one at ./data/Superstore.csv")
    st.stop()

# ---------- Data quality panel ----------
with st.expander("🔎 Data Quality Check"):
    st.write("Shape:", df.shape)
    st.dataframe(df.sample(min(len(df), 5)), use_container_width=True)
    st.write("Column types:", df.dtypes.to_frame("dtype"))
    st.write("Missing values (%):", (df.isna().mean() * 100).round(2).sort_values(ascending=False))

# ---------- Sidebar filters ----------
min_d, max_d = df["order_date"].min().date(), df["order_date"].max().date()
with st.sidebar:
    st.header("Filters")
    date_range = st.date_input("Order date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    regions = st.multiselect("Region", sorted(df["region"].dropna().unique().tolist()))
    segments = st.multiselect("Segment", sorted(df["segment"].dropna().unique().tolist()))

mask = (df["order_date"].dt.date >= date_range[0]) & (df["order_date"].dt.date <= date_range[1])
if regions:
    mask &= df["region"].isin(regions)
if segments:
    mask &= df["segment"].isin(segments)
df_f = df.loc[mask].copy()

if df_f.empty:
    st.warning("No data for the selected filters/date range. Try widening your filters.")
    st.stop()

# ----------KPIs ----------
total_revenue = float(df_f["sales"].sum())
total_profit  = float(df_f["profit"].sum())
margin_pct    = (total_profit / total_revenue) if total_revenue else 0.0
orders        = int(df_f["order_id"].nunique())
aov           = (total_revenue / orders) if orders else 0.0

gross_margin  = total_profit  # same as profit in this dataset; correct naming
unique_customers = df_f["customer_name"].nunique()
opc = orders / max(unique_customers, 1)  # orders per customer

weighted_discount = 0.0
if "discount" in df_f.columns:
    weighted_discount = (df_f["discount"] * df_f["sales"]).sum() / max(total_revenue, 1)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Revenue (£)", f"{total_revenue:,.0f}")
k2.metric("Profit (£)", f"{total_profit:,.0f}")
k3.metric("Margin %", f"{margin_pct:.1%}")
k4.metric("AOV (£)", f"{aov:,.0f}")
k5.metric("Orders / Customer", f"{opc:.2f}")
k6.metric("Weighted Discount %", f"{weighted_discount:.1%}")

st.divider()

# ---------- Monthly Revenue & Profit ----------
# ---------- Monthly Revenue & Profit (with 3-mo MA + naive forecast) ----------
df_f["order_month"] = df_f["order_date"].dt.to_period("M").dt.to_timestamp()
monthly = (df_f.groupby("order_month", as_index=False)
           .agg(revenue=("sales","sum"),
                profit=("profit","sum"),
                orders=("order_id","nunique")))

# Smooth the noise
monthly["rev_3mo_ma"] = monthly["revenue"].rolling(3).mean()

# Naive forecast: next month ≈ last 3-month average
monthly["rev_forecast"] = monthly["rev_3mo_ma"].shift(1)

st.subheader("Monthly Revenue & Profit")
st.line_chart(monthly.set_index("order_month")[["revenue","rev_3mo_ma","profit","rev_forecast"]])
# ---------- Category → Sub-Category ----------
st.subheader("Revenue by Category → Sub-Category")
cat = (df_f.groupby(["category","sub_category"], as_index=False)
       .agg(revenue=("sales","sum"), profit=("profit","sum"))
       .sort_values("revenue", ascending=False))
st.dataframe(cat, use_container_width=True)
# ---------- Top-N Products by Revenue ----------
st.subheader("Top-N Products by Revenue")
top_n = st.slider("Choose N", 5, 50, 15, step=5)

prod = (df_f.groupby("product_name", as_index=False)
          .agg(revenue=("sales","sum"),
               profit=("profit","sum"),
               orders=("order_id","nunique")))
prod["margin_pct"] = prod["profit"] / prod["revenue"]
top_table = prod.sort_values("revenue", ascending=False).head(top_n)

st.dataframe(top_table, use_container_width=True)
# ---------- Cohort Retention (customers) ----------
# ---------- Cohort Retention (count & rate) ----------
st.subheader("Customer Retention Cohorts (count & rate)")

# Cohort assignment from full dataset for stability
first_purchase = df.groupby("customer_name")["order_date"].min().dt.to_period("M")
df_f["cohort_month"] = df_f["customer_name"].map(first_purchase).astype("period[M]")
df_f["order_month_p"] = df_f["order_date"].dt.to_period("M")
df_f["months_since"] = (df_f["order_month_p"] - df_f["cohort_month"]).apply(attrgetter("n")).astype(int)

# Counts per cohort month and months since
cohort_counts = (df_f.groupby(["cohort_month","months_since"])["customer_name"]
                   .nunique()
                   .reset_index())

# Cohort sizes (month 0)
cohort_sizes = (cohort_counts[cohort_counts["months_since"].eq(0)]
                .loc[:, ["cohort_month","customer_name"]]
                .rename(columns={"customer_name":"cohort_size"}))

cohort = cohort_counts.merge(cohort_sizes, on="cohort_month", how="left")
cohort["retention_rate"] = cohort["customer_name"] / cohort["cohort_size"]

# Pivots
p_count = (cohort.pivot(index="cohort_month", columns="months_since", values="customer_name")
                 .fillna(0).astype(int))
p_rate  = (cohort.pivot(index="cohort_month", columns="months_since", values="retention_rate")
                 .fillna(0).clip(0,1))

# Heatmap (rate)
fig, ax = plt.subplots(figsize=(10,5))
im = ax.imshow(p_rate.values, aspect="auto", vmin=0, vmax=1)
ax.set_title("Retention Rate by Cohort")
ax.set_xlabel("Months Since Cohort")
ax.set_ylabel("Cohort Month")
ax.set_xticks(range(p_rate.shape[1])); ax.set_xticklabels(p_rate.columns)
ax.set_yticks(range(p_rate.shape[0])); ax.set_yticklabels([str(i) for i in p_rate.index])
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
st.pyplot(fig, clear_figure=True)

with st.expander("Tables"):
    # Arrow-friendly display
    p_count_disp = p_count.copy()
    p_count_disp.index = p_count_disp.index.astype(str)   # Period -> "YYYY-MM"
    st.write("Active customers (count)"); st.dataframe(p_count_disp, use_container_width=True)

    p_rate_disp = (p_rate*100).round(1)
    p_rate_disp.index = p_rate_disp.index.astype(str)
    st.write("Retention rate (%)"); st.dataframe(p_rate_disp, use_container_width=True)

# ---------- Auto Insights ----------
st.subheader("Auto Insights")

insights = []

# Top category by margin £
cat_perf = (df_f.groupby("category", as_index=False)
              .agg(revenue=("sales","sum"), profit=("profit","sum")))
if not cat_perf.empty:
    top_cat = cat_perf.sort_values("profit", ascending=False).iloc[0]
    insights.append(f"Top category by margin £ is **{top_cat['category']}** (£{top_cat['profit']:,.0f}).")

# Worst sub-category by margin £
sub_perf = (df_f.groupby("sub_category", as_index=False)
              .agg(revenue=("sales","sum"), profit=("profit","sum")))
if not sub_perf.empty:
    worst_sub = sub_perf.sort_values("profit", ascending=True).iloc[0]
    insights.append(f"Weakest sub-category by margin £ is **{worst_sub['sub_category']}** (-£{abs(worst_sub['profit']):,.0f}).")

# Peak AOV month
if not monthly.empty:
    monthly["aov"] = monthly["revenue"] / monthly["orders"].replace(0, np.nan)
    m = monthly.dropna(subset=["aov"]).sort_values("aov", ascending=False).iloc[0]
    insights.append(f"Highest AOV month: **{m['order_month'].date()}** (AOV £{m['aov']:,.0f}).")

if insights:
    for s in insights:
        st.markdown(f"- {s}")
else:
    st.info("No insights available for the current filters.")
