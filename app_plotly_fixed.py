# =============================================================
# Sales Funnel Executive Dashboard v4 - Enhanced with Funnel Velocity
# FIXED VERSION - Customer-Level Funnel Logic
# =============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def css_background_gradient(s, color_low=(255,255,255), color_high=(49,130,206)):
    """Apply a CSS background gradient to a pandas Series without matplotlib."""
    if s.isna().all() or s.min() == s.max():
        return [''] * len(s)
    normalized = (s - s.min()) / (s.max() - s.min())
    styles = []
    for v in normalized:
        if pd.isna(v):
            styles.append('')
        else:
            r = int(color_low[0] + (color_high[0] - color_low[0]) * v)
            g = int(color_low[1] + (color_high[1] - color_low[1]) * v)
            b = int(color_low[2] + (color_high[2] - color_low[2]) * v)
            styles.append(f'background-color: rgb({r},{g},{b}); color: {"white" if v > 0.6 else "black"}')
    return styles

# Color presets (low → high) to replace matplotlib cmaps
GRADIENT_COLORS = {
    "RdYlGn": ((255, 100, 100), (100, 200, 100)),
    "Blues":  ((235, 245, 255), (49, 130, 206)),
    "Greens": ((235, 255, 235), (56, 161, 105)),
    "Purples":((240, 235, 255), (128, 90, 213)),
    "Reds":   ((255, 240, 240), (220, 80, 80)),
}


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Sales Funnel Executive Dashboard v4.1",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Sales Funnel Executive Dashboard v4.1 - Behavioral Analytics & Sequential Funnel Logic"
    }
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
    .filter-badge {
        background-color: #3182ce;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        margin-right: 8px;
        display: inline-block;
    }
    .metric-card {
        background-color: #f7fafc;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #3182ce;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CSV Path / Upload Option
# -----------------------------

@st.cache_data(show_spinner="Loading CSV...")
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data(show_spinner="Preparing data...")
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # put your expensive cleaning + feature engineering here
    return df

df_raw = load_csv("sales_funnel_dataset.csv")
df = prepare(df_raw)


# =============================================================
# DATA LOADING AND PREPARATION
# =============================================================
REQUIRED_COLS = [
    "customer_id", "event_type", "product_view_at", "add_to_cart_at", 
    "checkout_at", "purchase_at", "Order Date", "Expected Delivery Date",
    "sales_rep", "signup_date", "acquisition", "country", "List Price", 
    "Discount", "Final Price"
]

def build_customer_journey(df):
    """Combine event rows into one row per customer"""
    
    journey = df.groupby("customer_id").agg({
        "product_view_at": "max",
        "add_to_cart_at": "max",
        "checkout_at": "max",
        "purchase_at": "max",
        "Order Date": "max",
        "Expected Delivery Date": "max",
        "signup_date": "first",
        "sales_rep": "first",
        "acquisition": "first",
        "country": "first",
        "List Price": "max",
        "Discount": "max",
        "Final Price": "max"
    }).reset_index()
    
    return journey

def prepare_data(df):
    """Enhanced data preparation with funnel velocity calculations"""
    df = df.copy()
    
    # Convert timestamps
    timestamp_cols = ["product_view_at", "add_to_cart_at", "checkout_at", 
                     "purchase_at", "Order Date", "Expected Delivery Date", "signup_date"]
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Generate missing IDs
    df["order_id"] = [f"ORD_{i:06d}" for i in range(len(df))]
    df["event_id"] = [f"EVT_{i:06d}" for i in range(len(df))]
    
    # Deal status based on realistic pipeline behavior
    today = pd.Timestamp.now()
    
    df["deal_status"] = np.select(
        [
            df["purchase_at"].notna(),
            (df["checkout_at"].notna()) & (df["purchase_at"].isna()) & ((today - df["checkout_at"]).dt.days > 3),
            (df["product_view_at"].notna()) & (df["checkout_at"].isna())
        ],
        [
            "Won",
            "Lost",
            "In Progress"
        ],
        default="New"
    )
    
    # Fill Order Date for purchases from purchase_at
    df.loc[df["Order Date"].isna() & df["purchase_at"].notna(), "Order Date"] = \
        df.loc[df["Order Date"].isna() & df["purchase_at"].notna(), "purchase_at"]
    
    # For non-purchases, use the last known timestamp as Order Date
    df.loc[df["Order Date"].isna(), "Order Date"] = df.loc[df["Order Date"].isna()].apply(
        lambda row: row[["checkout_at", "add_to_cart_at", "product_view_at"]].dropna().max() 
        if not row[["checkout_at", "add_to_cart_at", "product_view_at"]].isna().all() 
        else pd.NaT, axis=1
    )
    
    # Calculate funnel velocity metrics (time between stages in hours)
    df["view_to_cart_hours"] = (df["add_to_cart_at"] - df["product_view_at"]).dt.total_seconds() / 3600
    df["cart_to_checkout_hours"] = (df["checkout_at"] - df["add_to_cart_at"]).dt.total_seconds() / 3600
    df["checkout_to_purchase_hours"] = (df["purchase_at"] - df["checkout_at"]).dt.total_seconds() / 3600
    df["total_funnel_hours"] = (df["purchase_at"] - df["product_view_at"]).dt.total_seconds() / 3600
    
    # Pricing should only exist for purchases
    df.loc[df["purchase_at"].isna(), ["List Price", "Discount", "Final Price"]] = np.nan
    
    # Revenue and pricing calculations
    df["expected_price"] = df["List Price"]
    df["revenue_loss"] = df["expected_price"] - df["Final Price"]
    df["days_to_delivery"] = (df["Expected Delivery Date"] - df["Order Date"]).dt.days
    df["customer_age_days"] = (df["Order Date"] - df["signup_date"]).dt.days
    
    # Discount rate (only for purchases with pricing)
    df["discount_rate"] = np.where(
        df["List Price"] > 0,
        (df["Discount"] / df["List Price"] * 100),
        0
    ).round(2)
    
    # High discount flag (only for records with discount data)
    threshold = df[df["Discount"] > 0]["Discount"].quantile(0.90) if (df["Discount"] > 0).any() else 0
    df["high_discount_flag"] = np.where(df["Discount"] > threshold, 1, 0)
    
    # Optimized pricing
    df["optimized_price"] = df["expected_price"] * 1.05
    df["recovered_revenue"] = (df["optimized_price"] - df["Final Price"]).clip(lower=0)
    
    # Time periods
    df["order_month"] = df["Order Date"].dt.to_period("M")
    df["order_quarter"] = df["Order Date"].dt.to_period("Q")
    df["order_week"] = df["Order Date"].dt.to_period("W")
    
    return df


# Load CSV
try:
    raw_df = pd.read_csv(CSV_PATH)
    st.sidebar.success(f"✅ Loaded {len(raw_df)} records")
except FileNotFoundError:
    st.error(f"❌ CSV file not found at: {CSV_PATH}")
    st.info("Please update the CSV_PATH variable in the code or upload your file.")
    st.stop()

# Convert event data → customer journey data
journey_df = build_customer_journey(raw_df)
df = prepare_data(journey_df)


# =============================================================
# SIDEBAR – FILTERS
# =============================================================
st.sidebar.header("🎯 Filters")

total_records = len(df)

# --- Date range ---
st.sidebar.subheader("📅 Date Range")
date_preset = st.sidebar.radio(
    "Quick Select:",
    ["All Time", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Year to Date", "Custom"],
    horizontal=False, label_visibility="collapsed"
)

min_date = df["Order Date"].min()
max_date = df["Order Date"].max()

if date_preset == "Last 30 Days":
    start_date, end_date = max_date - timedelta(days=30), max_date
elif date_preset == "Last 90 Days":
    start_date, end_date = max_date - timedelta(days=90), max_date
elif date_preset == "Last 6 Months":
    start_date, end_date = max_date - timedelta(days=180), max_date
elif date_preset == "Year to Date":
    start_date, end_date = pd.Timestamp(max_date.year, 1, 1), max_date
elif date_preset == "Custom":
    date_range = st.sidebar.date_input(
        "Select Date Range:", value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )
    start_date, end_date = (date_range if len(date_range) == 2 else (min_date, max_date))
else:  # All Time
    start_date, end_date = min_date, max_date

st.sidebar.divider()

# --- Deal Status ---
st.sidebar.subheader("📊 Deal Status")
status_counts_all = df["deal_status"].value_counts()
status_options = ["All"] + [f"{s} ({status_counts_all.get(s, 0)})" for s in df["deal_status"].unique()]
sel_status_disp = st.sidebar.selectbox("Status:", status_options, key="status_filter")
selected_status = sel_status_disp.split(" (")[0] if sel_status_disp != "All" else "All"

st.sidebar.divider()

# --- Country ---
st.sidebar.subheader("🌍 Country")
country_options = ["All"] + sorted(df["country"].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Country:", country_options, key="country_filter")

st.sidebar.divider()

# --- Sales Rep ---
st.sidebar.subheader("👤 Sales Rep")
rep_options = ["All"] + sorted(df["sales_rep"].dropna().unique().tolist())
selected_rep = st.sidebar.selectbox("Rep:", rep_options, key="rep_filter")

st.sidebar.divider()

# --- Acquisition Channel ---
st.sidebar.subheader("📢 Acquisition Channel")
acq_options = ["All"] + sorted(df["acquisition"].dropna().unique().tolist())
selected_channel = st.sidebar.selectbox("Channel:", acq_options, key="channel_filter")

# =============================================================
# APPLY FILTERS
# =============================================================
filtered_df = df.copy()

# Date range
filtered_df = filtered_df[(filtered_df["Order Date"] >= pd.Timestamp(start_date)) &
                          (filtered_df["Order Date"] <= pd.Timestamp(end_date))]

# Deal status
if selected_status != "All":
    filtered_df = filtered_df[filtered_df["deal_status"] == selected_status]

# Country
if selected_country != "All":
    filtered_df = filtered_df[filtered_df["country"] == selected_country]

# Rep
if selected_rep != "All":
    filtered_df = filtered_df[filtered_df["sales_rep"] == selected_rep]

# Channel
if selected_channel != "All":
    filtered_df = filtered_df[filtered_df["acquisition"] == selected_channel]

# Display active filters
active_filters = []
if date_preset != "All Time":
    active_filters.append(f"📅 {date_preset}")
if selected_status != "All":
    active_filters.append(f"📊 {selected_status}")
if selected_country != "All":
    active_filters.append(f"🌍 {selected_country}")
if selected_rep != "All":
    active_filters.append(f"👤 {selected_rep}")
if selected_channel != "All":
    active_filters.append(f"📢 {selected_channel}")

if active_filters:
    st.sidebar.markdown("**Active Filters:**")
    for f in active_filters:
        st.sidebar.markdown(f'<span class="filter-badge">{f}</span>', unsafe_allow_html=True)

st.sidebar.markdown(f"**Records:** {len(filtered_df):,} / {total_records:,}")

# =============================================================
# HEADER
# =============================================================
st.title("📊 Sales Funnel Executive Dashboard v4.1")
st.markdown("**Sequential funnel logic • Behavioral insights • Checkout abandonment analysis • Speed segmentation**")
st.divider()

# =============================================================
# §1 KEY METRICS (CUSTOMER-LEVEL)
# =============================================================

# Sequential funnel logic (correct funnel behavior)
views = filtered_df["product_view_at"].notna()

carts = views & filtered_df["add_to_cart_at"].notna()
checkouts = carts & filtered_df["checkout_at"].notna()
purchases = checkouts & filtered_df["purchase_at"].notna()

views = views.sum()
carts = carts.sum()
checkouts = checkouts.sum()
purchases = purchases.sum()

unique_visitors = views
unique_buyers = purchases

# Revenue metrics
total_revenue = filtered_df.loc[filtered_df["purchase_at"].notna(), "Final Price"].sum()
total_discount = filtered_df.loc[filtered_df["purchase_at"].notna(), "Discount"].sum()
total_purchases = purchases

overall_conversion_rate = (unique_buyers / unique_visitors * 100) if unique_visitors > 0 else 0


# Display KPIs
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("👥 Unique Visitors", f"{unique_visitors:,}",
            help="Unique customers who viewed products")
col2.metric("🛒 In Cart", f"{carts:,}",
            help="Customers who added to cart")
col3.metric("💳 In Checkout", f"{checkouts:,}",
            help="Customers who reached checkout")
col4.metric("✅ Purchases", f"{purchases:,}",
            help="Customers who completed purchase")
col5.metric("📈 Conversion Rate", f"{overall_conversion_rate:.1f}%",
            delta=f"{overall_conversion_rate - 5:.1f}% vs target",
            help="Unique buyers / unique visitors")

st.divider()

# =============================================================
# §2 REVENUE METRICS
# =============================================================
st.header("💰 Revenue Overview")

rcol1, rcol2, rcol3, rcol4 = st.columns(4)

avg_purchase_value = total_revenue / total_purchases if total_purchases > 0 else 0
avg_discount_rate = (total_discount / (total_revenue + total_discount) * 100) if (total_revenue + total_discount) > 0 else 0

rcol1.metric("Total Revenue", f"${total_revenue:,.0f}")
rcol2.metric("Avg Deal Value", f"${avg_purchase_value:,.0f}")
rcol3.metric("Total Discounts", f"${total_discount:,.0f}")
rcol4.metric("Avg Discount Rate", f"{avg_discount_rate:.1f}%")

st.divider()

# =============================================================
# §3 FUNNEL VISUALIZATION (FIX #2)
# =============================================================
st.header("🔀 Sales Funnel Breakdown")

funnel_counts = {
    "Product_View": views,
    "Add_to_Cart": carts,
    "Checkout": checkouts,
    "Purchase": purchases
}

funnel_data = pd.DataFrame(list(funnel_counts.items()), columns=["Stage", "Count"])
funnel_data["Percentage"] = (funnel_data["Count"] / funnel_data["Count"].iloc[0] * 100).round(1)

fig = go.Figure(go.Funnel(
    y=funnel_data["Stage"],
    x=funnel_data["Count"],
    textposition="inside",
    textinfo="value+percent initial",
    marker={"color": ["#3182ce", "#38a169", "#ed8936", "#9f7aea"]},
    connector={"line": {"color": "#cbd5e0", "width": 2}}
))

fig.update_layout(
    height=400,
    margin=dict(l=150, r=50, t=50, b=50),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig, use_container_width=True)

# Stage-by-stage conversion
st.subheader("📊 Stage-by-Stage Conversion Rates")
funnel_metrics = pd.DataFrame({
    "Stage": ["View → Cart", "Cart → Checkout", "Checkout → Purchase"],
    "Conversion %": [
        (carts / views * 100) if views > 0 else 0,
        (checkouts / carts * 100) if carts > 0 else 0,
        (purchases / checkouts * 100) if checkouts > 0 else 0
    ]
})

fcol1, fcol2, fcol3 = st.columns(3)
for i, row in funnel_metrics.iterrows():
    col = [fcol1, fcol2, fcol3][i]
    col.metric(row["Stage"], f"{row['Conversion %']:.1f}%")

# Checkout Abandonment Analysis
st.subheader("🚨 Checkout Abandonment Insight")

checkout_abandoned = ((filtered_df["checkout_at"].notna()) &
                      (filtered_df["purchase_at"].isna())).sum()

checkout_started = filtered_df["checkout_at"].notna().sum()

abandonment_rate = (checkout_abandoned / checkout_started * 100) if checkout_started > 0 else 0

abcol1, abcol2, abcol3 = st.columns(3)
abcol1.metric(
    "Checkout Abandonment Rate",
    f"{abandonment_rate:.1f}%",
    help="Customers who reached checkout but did not complete payment"
)
abcol2.metric(
    "Abandoned Checkouts",
    f"{checkout_abandoned:,}",
    help="Total customers who started but didn't complete checkout"
)
abcol3.metric(
    "Completed Checkouts",
    f"{purchases:,}",
    help="Customers who successfully purchased"
)

st.info("""
💡 **High abandonment?** Common causes:
- Payment friction or limited payment methods
- Unexpected shipping costs
- Trust/security concerns
- Complex checkout process
- Page load issues
""")

st.divider()

# =============================================================
# §4 FUNNEL VELOCITY ANALYSIS
# =============================================================
st.header("⚡ Funnel Velocity Analysis")
st.markdown("*Time customers spend at each funnel stage*")

# Calculate velocity metrics for purchases only
purchases_df = filtered_df[filtered_df["purchase_at"].notna()].copy()

if len(purchases_df) > 0:
    avg_view_to_cart = purchases_df["view_to_cart_hours"].median()
    avg_cart_to_checkout = purchases_df["cart_to_checkout_hours"].median()
    avg_checkout_to_purchase = purchases_df["checkout_to_purchase_hours"].median()
    avg_total_funnel = purchases_df["total_funnel_hours"].median()
    
    vcol1, vcol2, vcol3, vcol4 = st.columns(4)
    vcol1.metric("View → Cart", f"{avg_view_to_cart:.1f}h", help="Median time from view to cart")
    vcol2.metric("Cart → Checkout", f"{avg_cart_to_checkout:.1f}h", help="Median time from cart to checkout")
    vcol3.metric("Checkout → Purchase", f"{avg_checkout_to_purchase:.1f}h", help="Median time from checkout to purchase")
    vcol4.metric("Total Funnel Time", f"{avg_total_funnel:.1f}h", help="Median total time from view to purchase")
    
    # Velocity distribution chart
    st.subheader("Distribution of Funnel Times")
    
    velocity_data = pd.DataFrame({
        "Stage": ["View→Cart"] * len(purchases_df) + ["Cart→Checkout"] * len(purchases_df) + ["Checkout→Purchase"] * len(purchases_df),
        "Hours": purchases_df["view_to_cart_hours"].tolist() + purchases_df["cart_to_checkout_hours"].tolist() + purchases_df["checkout_to_purchase_hours"].tolist()
    })
    
    # Remove outliers for better visualization (keep 5-95 percentile)
    velocity_data = velocity_data[
        (velocity_data["Hours"] >= velocity_data["Hours"].quantile(0.05)) &
        (velocity_data["Hours"] <= velocity_data["Hours"].quantile(0.95))
    ]
    
    fig = px.box(velocity_data, x="Stage", y="Hours", color="Stage",
                 color_discrete_map={"View→Cart": "#3182ce", "Cart→Checkout": "#38a169", "Checkout→Purchase": "#ed8936"})
    fig.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=30, b=40),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)
    
    # ---------------- Velocity Segmentation ----------------
    st.subheader("🧠 Buyer Speed vs Revenue Insight")
    
    if len(purchases_df) > 10:
        # Create speed segments based on total funnel time
        purchases_df["speed_segment"] = pd.qcut(
            purchases_df["total_funnel_hours"],
            q=3,
            labels=["Fast Buyers", "Medium Buyers", "Slow Buyers"],
            duplicates='drop'
        )
        
        speed_analysis = purchases_df.groupby("speed_segment", observed=True).agg({
            "Final Price": ["mean", "sum", "count"],
            "total_funnel_hours": "median",
            "discount_rate": "mean"
        }).round(2)
        
        speed_analysis.columns = ["Avg Deal Value", "Revenue", "Purchases", "Median Hours", "Avg Discount %"]
        
        # Display table WITHOUT pandas Styler (avoids matplotlib dependency in Streamlit Cloud)
        speed_table = speed_analysis.copy()

        # Make sure numeric columns are numeric
        for _c in ["Avg Deal Value", "Revenue", "Purchases", "Median Hours", "Avg Discount %"]:
            if _c in speed_table.columns:
                speed_table[_c] = pd.to_numeric(speed_table[_c], errors="coerce")

        # Create a string-formatted display version (keeps sorting/logic in speed_table)
        speed_table_display = speed_table.copy()
        if "Avg Deal Value" in speed_table_display.columns:
            speed_table_display["Avg Deal Value"] = speed_table_display["Avg Deal Value"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "")
        if "Revenue" in speed_table_display.columns:
            speed_table_display["Revenue"] = speed_table_display["Revenue"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "")
        if "Purchases" in speed_table_display.columns:
            speed_table_display["Purchases"] = speed_table_display["Purchases"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
        if "Median Hours" in speed_table_display.columns:
            speed_table_display["Median Hours"] = speed_table_display["Median Hours"].map(lambda v: f"{v:.1f}h" if pd.notna(v) else "")
        if "Avg Discount %" in speed_table_display.columns:
            speed_table_display["Avg Discount %"] = speed_table_display["Avg Discount %"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")

        st.dataframe(
            speed_table_display,
            use_container_width=True
        )

        # Plotly: Revenue by speed segment (replaces pandas Styler gradient)
        fig = go.Figure(go.Bar(
            x=speed_table.index.astype(str).tolist(),
            y=speed_table["Revenue"].tolist(),
            text=[f"${v:,.0f}" if pd.notna(v) else "" for v in speed_table["Revenue"]],
            textposition="auto",
            marker_color="#3182ce"
        ))
        fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=30, b=40),
            xaxis_title="Speed Segment",
            yaxis_title="Revenue ($)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        fig.update_yaxes(gridcolor="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

        
        # Insight interpretation
        fast_rev = speed_analysis.loc["Fast Buyers", "Avg Deal Value"]
        slow_rev = speed_analysis.loc["Slow Buyers", "Avg Deal Value"]
        
        if fast_rev > slow_rev * 1.1:
            st.success("""
            ✅ **Key Insight:** Fast buyers generate higher revenue! 
            - **Action:** Reduce friction in checkout, add urgency triggers, streamline the funnel
            - **Strategy:** Hesitation = Risk. Speed up decision-making with limited offers
            """)
        elif slow_rev > fast_rev * 1.1:
            st.info("""
            💡 **Key Insight:** Slow buyers are more valuable (evaluation phase).
            - **Action:** Provide detailed info, comparison tools, expert consultations
            - **Strategy:** These are considered purchases, not impulse buys
            """)
        else:
            st.info("📊 **Insight:** Speed has neutral impact on deal value. Focus on conversion optimization across all segments.")
    else:
        st.info("Need at least 10 purchases for speed segmentation analysis.")
        
else:
    st.info("No purchase data available for velocity analysis.")
    avg_total_funnel = 0

st.divider()

# =============================================================
# §5 REP PERFORMANCE
# =============================================================
st.header("👥 Sales Rep Performance")

rep_perf = filtered_df[filtered_df["purchase_at"].notna()].groupby("sales_rep").agg({
    "Final Price": ["sum", "mean", "count"],
    "Discount": "mean"
}).round(2)

rep_perf.columns = ["Revenue", "Avg_Deal", "Deals", "Avg_Discount"]
rep_perf = rep_perf.sort_values("Revenue", ascending=False)

if len(rep_perf) > 0:
    top_rep = rep_perf.index[0]
    
    # Display WITHOUT pandas Styler (avoids matplotlib dependency)
    rep_perf_display = rep_perf.reset_index().rename(columns={"index": "Sales Rep"})
    rep_perf_display["Revenue"] = rep_perf_display["Revenue"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "")
    rep_perf_display["Avg_Deal"] = rep_perf_display["Avg_Deal"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "")
    rep_perf_display["Deals"] = rep_perf_display["Deals"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
    rep_perf_display["Avg_Discount"] = rep_perf_display["Avg_Discount"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "")
    st.dataframe(rep_perf_display, use_container_width=True)

    
    # Rep comparison chart
    fig = go.Figure(go.Bar(
        x=rep_perf.index.tolist(),
        y=rep_perf["Revenue"].tolist(),
        text=[f"${v:,.0f}" for v in rep_perf["Revenue"]],
        textposition="auto",
        marker_color="#3182ce"
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=80),
                      xaxis_title="Sales Rep", yaxis_title="Revenue ($)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    fig.update_yaxes(gridcolor="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No sales rep data available.")
    top_rep = "N/A"

st.divider()

# =============================================================
# §6 CHANNEL ANALYSIS
# =============================================================
st.header("📢 Acquisition Channel Performance")

# Calculate channel metrics based on unique customers
channel_views = filtered_df[filtered_df["product_view_at"].notna()].groupby("acquisition")["customer_id"].nunique()
channel_purchases = filtered_df[filtered_df["purchase_at"].notna()].groupby("acquisition")["customer_id"].nunique()
channel_revenue = filtered_df[filtered_df["purchase_at"].notna()].groupby("acquisition")["Final Price"].sum()

ch_analysis = pd.DataFrame({
    "Views": channel_views,
    "Purchases": channel_purchases,
    "Revenue": channel_revenue
}).fillna(0)

ch_analysis["Conversion_Rate"] = (ch_analysis["Purchases"] / ch_analysis["Views"] * 100).round(2)
ch_analysis["Avg_Deal"] = (ch_analysis["Revenue"] / ch_analysis["Purchases"]).fillna(0).round(2)
ch_analysis = ch_analysis.sort_values("Revenue", ascending=False)

if len(ch_analysis) > 0:
    top_channel = ch_analysis.index[0]
    
    # Display WITHOUT pandas Styler (avoids matplotlib dependency)
    ch_display = ch_analysis.reset_index().rename(columns={"index": "Channel"})
    for _c in ["Views", "Purchases"]:
        if _c in ch_display.columns:
            ch_display[_c] = ch_display[_c].map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
    if "Revenue" in ch_display.columns:
        ch_display["Revenue"] = ch_display["Revenue"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "")
    if "Conversion_Rate" in ch_display.columns:
        ch_display["Conversion_Rate"] = ch_display["Conversion_Rate"].map(lambda v: f"{v:.2f}%" if pd.notna(v) else "")
    if "Avg_Deal" in ch_display.columns:
        ch_display["Avg_Deal"] = ch_display["Avg_Deal"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "")
    st.dataframe(ch_display, use_container_width=True)

    
    # Channel revenue chart
    fig = go.Figure(go.Bar(
        x=ch_analysis.index.tolist(),
        y=ch_analysis["Revenue"].tolist(),
        text=[f"${v:,.0f}" for v in ch_analysis["Revenue"]],
        textposition="auto",
        marker_color="#38a169"
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=80),
                      xaxis_title="Channel", yaxis_title="Revenue ($)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    fig.update_yaxes(gridcolor="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No channel data available.")
    top_channel = "N/A"

st.divider()

# =============================================================
# §7 COUNTRY PERFORMANCE
# =============================================================
st.header("🌍 Geographic Performance")

country_perf = filtered_df[filtered_df["purchase_at"].notna()].groupby("country").agg({
    "Final Price": ["sum", "count"],
    "customer_id": "nunique"
}).round(2)

country_perf.columns = ["Revenue", "Deals", "Unique_Customers"]
country_perf = country_perf.sort_values("Revenue", ascending=False).head(10)

if len(country_perf) > 0:
    top_country = country_perf.index[0]
    
    # Display WITHOUT pandas Styler (avoids matplotlib dependency)
    country_display = country_perf.reset_index().rename(columns={"index": "Country"})
    if "Revenue" in country_display.columns:
        country_display["Revenue"] = country_display["Revenue"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "")
    for _c in ["Deals", "Unique_Customers"]:
        if _c in country_display.columns:
            country_display[_c] = country_display[_c].map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
    st.dataframe(country_display, use_container_width=True)

else:
    st.info("No country data available.")
    top_country = "N/A"

st.divider()

# =============================================================
# §8 REVENUE & DISCOUNT ANALYSIS
# =============================================================
st.header("💵 Revenue & Discount Analysis")

if len(purchases_df) > 0:
    cr1, cr2 = st.columns(2)
    
    with cr1:
        st.subheader("Revenue Trend")
        revenue_trend = purchases_df.groupby(purchases_df["Order Date"].dt.to_period("M"))["Final Price"].sum().reset_index()
        revenue_trend["Order Date"] = revenue_trend["Order Date"].astype(str)
        
        fig = go.Figure(go.Scatter(
            x=revenue_trend["Order Date"], y=revenue_trend["Final Price"],
            mode="lines+markers", line=dict(color="#3182ce", width=3),
            marker=dict(size=8), fill="tozeroy", fillcolor="rgba(49, 130, 206, 0.1)"
        ))
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=40),
                          xaxis_title="Month", yaxis_title="Revenue ($)",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_yaxes(gridcolor="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)
    
    with cr2:
        st.subheader("Discount Distribution")
        fig = go.Figure(go.Histogram(
            x=purchases_df["discount_rate"],
            nbinsx=20,
            marker_color="#805ad5"
        ))
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=40),
                          xaxis_title="Discount Rate (%)", yaxis_title="Count",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          showlegend=False)
        fig.update_yaxes(gridcolor="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)
    
    # High discount analysis
    st.subheader("💸 High Discount Impact")
    high_disc_purchases = purchases_df[purchases_df["high_discount_flag"] == 1]
    normal_disc_purchases = purchases_df[purchases_df["high_discount_flag"] == 0]
    
    disc_impact = pd.DataFrame({
        "Category": ["Normal Discounts", "High Discounts (Top 10%)"],
        "Count": [len(normal_disc_purchases), len(high_disc_purchases)],
        "Revenue": [normal_disc_purchases["Final Price"].sum(), high_disc_purchases["Final Price"].sum()],
        "Avg Discount": [normal_disc_purchases["Discount"].mean(), high_disc_purchases["Discount"].mean()]
    })
    
    # Display WITHOUT pandas Styler (avoids matplotlib dependency)
    disc_display = disc_impact.copy()
    if "Count" in disc_display.columns:
        disc_display["Count"] = disc_display["Count"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "")
    if "Revenue" in disc_display.columns:
        disc_display["Revenue"] = disc_display["Revenue"].map(lambda v: f"${v:,.0f}" if pd.notna(v) else "")
    if "Avg Discount" in disc_display.columns:
        disc_display["Avg Discount"] = disc_display["Avg Discount"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "")
    st.dataframe(disc_display, use_container_width=True)

else:
    st.info("No purchase data available for revenue analysis.")

st.divider()

# =============================================================
# §9 SCENARIO PLANNING TOOL (FIX #4)
# =============================================================
st.header("🎛️ Scenario Planning Tool")
st.markdown("Model potential revenue impact from operational improvements.")

sp1, sp2, sp3 = st.columns(3)
with sp1:
    conv_improve = st.slider("📈 Conversion Rate Improvement (%)", 0, 50, 10)
with sp2:
    disc_reduce = st.slider("💵 Discount Reduction (%)", 0, 50, 20)
with sp3:
    vel_improve = st.slider("⚡ Funnel Velocity Improvement (%)", 0, 50, 15,
                            help="Reduce time in funnel to increase throughput")

# Calculate impacts - BEHAVIOR-BASED PROJECTION
# Base projection on high-intent buyers (checkout users) instead of all visitors
checkout_users = filtered_df["checkout_at"].notna().sum()

current_checkout_conversion = (
    purchases / checkout_users * 100
    if checkout_users > 0 else 0
)

improved_checkout_conversion = min(current_checkout_conversion + conv_improve, 100)

new_purchases = int(checkout_users * (improved_checkout_conversion / 100))

# Average purchase value stays the same
avg_purchase_value = total_revenue / total_purchases if total_purchases > 0 else 0

# Projected revenue and conversion impact
projected_revenue = new_purchases * avg_purchase_value
conv_impact = projected_revenue - total_revenue

disc_impact = total_discount * (disc_reduce / 100)

# Velocity impact (assume faster funnel = more conversions)
vel_impact = total_revenue * (vel_improve / 100) * 0.5  # Conservative estimate

total_impact = conv_impact + disc_impact + vel_impact

# Display current vs projected metrics
st.markdown(f"""
**Current Metrics:**
- Checkout Users: **{checkout_users:,}**
- Checkout → Purchase Rate: **{current_checkout_conversion:.1f}%**
- Projected Rate: **{improved_checkout_conversion:.1f}%**
""")

si1, si2, si3, si4 = st.columns(4)
si1.metric("Conversion Impact", f"${conv_impact:,.0f}", delta=f"+{new_purchases - total_purchases} purchases")
si2.metric("Discount Savings", f"${disc_impact:,.0f}", delta=f"-{disc_reduce}% disc")
si3.metric("Velocity Impact", f"${vel_impact:,.0f}", delta=f"+{vel_improve}% faster")
si4.metric("Total Revenue Gain", f"${total_impact:,.0f}",
           delta=f"+{total_impact/total_revenue*100:.1f}%" if total_revenue > 0 else None)

# Impact breakdown chart
impact_df = pd.DataFrame({
    "Category": ["Conversion", "Discount", "Velocity"],
    "Impact": [conv_impact, disc_impact, vel_impact]
})
fig = go.Figure(go.Bar(
    x=impact_df["Category"].tolist(), y=impact_df["Impact"].tolist(),
    text=[f"${v:,.0f}" for v in impact_df["Impact"]], textposition="auto",
    marker_color=["#3182ce", "#38a169", "#ed8936"]
))
fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=40),
                  yaxis_title="Revenue Impact ($)",
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
fig.update_yaxes(gridcolor="#e2e8f0")
st.plotly_chart(fig, use_container_width=True)

st.success(
    f"✅ **Projected Gain: ${total_impact:,.0f}** "
    f"({total_impact/total_revenue*100:.1f}% over current revenue). "
    "\n\n**This model targets high-intent buyers (checkout stage)** - a realistic operational projection."
)
st.info("""
💡 **Model Methodology:** 
- Focuses on customers who reached checkout (high purchase intent)
- Assumes improvements in checkout friction, payment options, and trust signals
- More realistic than modeling all visitors (includes browsers with no intent)
""")
st.divider()

# =============================================================
# §10 EXECUTIVE INSIGHTS & RECOMMENDATIONS
# =============================================================
st.header("🧠 Executive Insights & Recommendations")

ins1, ins2 = st.columns(2)

with ins1:
    st.subheader("✅ Key Findings")
    
    # Identify bottleneck stage
    bottleneck_stage = funnel_metrics.loc[funnel_metrics["Conversion %"].idxmin(), "Stage"] if len(funnel_metrics) > 1 else "N/A"
    lowest_conversion = funnel_metrics["Conversion %"].min() if len(funnel_metrics) > 1 else 0
    
    st.markdown(f"""
**Funnel Performance**
- 🎯 Overall Conversion: **{overall_conversion_rate:.1f}%**
- 📊 Total Purchases: **{purchases}** out of **{unique_visitors}** unique visitors
- ⚠️ Biggest Bottleneck: **{bottleneck_stage}** ({lowest_conversion:.1f}% conversion)
- 💰 Total Revenue: **${total_revenue:,.0f}**

**Top Performers**
- 🏆 Best Rep: **{top_rep}** (${rep_perf.loc[top_rep, "Revenue"]:,.0f})
- 📢 Best Channel: **{top_channel}** ({ch_analysis.loc[top_channel, "Conversion_Rate"]:.1f}% conversion)
- 🌍 Top Country: **{top_country}**

**Revenue Metrics**
- 💵 Avg Discount: **{avg_discount_rate:.1f}%** (**${total_discount:,.0f}** total)
- 💸 Avg Deal Value: **${avg_purchase_value:,.0f}**
""")

with ins2:
    st.subheader("🎯 Strategic Recommendations")
    st.markdown(f"""
**1. Address Funnel Bottleneck**
- Focus on improving **{bottleneck_stage}** conversion (currently {lowest_conversion:.1f}%)
- Implement automated follow-ups and nurture campaigns
- A/B test messaging and incentives at this stage

**2. Optimize Funnel Velocity**
- Reduce time between stages (current avg: {avg_total_funnel:.1f}h total)
- Implement urgency triggers and limited-time offers
- Streamline checkout process and reduce friction

**3. Channel & Rep Optimization**
- Scale investment in **{top_channel}** (best conversion)
- Replicate **{top_rep}'s** strategies across team
- Provide targeted coaching for underperforming reps

**4. Discount Management**
- Monitor high-discount deals ({len(purchases_df[purchases_df["high_discount_flag"]==1])} in top 10%)
- Implement approval workflows for discounts >20%
- Test value-based pricing vs. discount-driven sales

**5. Quick Wins**
- Re-engage abandoned carts within 24 hours
- Implement exit-intent popups for Product_View stage
- Create urgency messaging for Checkout stage
""")

st.divider()

# =============================================================
# §11 DATA EXPLORER
# =============================================================
with st.expander("🔍 Detailed Data Explorer", expanded=False):
    st.markdown(f"Showing **{len(filtered_df):,}** records")
    default_cols = ["Order Date", "customer_id", "deal_status", "sales_rep",
                    "Final Price", "Discount", "country", "acquisition",
                    "product_view_at", "add_to_cart_at", "checkout_at", "purchase_at"]
    sel_cols = st.multiselect("Columns:", filtered_df.columns.tolist(), default=default_cols)
    if sel_cols:
        st.dataframe(filtered_df[sel_cols].reset_index(drop=True), use_container_width=True, height=450)
    else:
        st.info("Select at least one column.")

st.divider()

# =============================================================
# FOOTER
# =============================================================
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("**📊 Version:** 4.1 - Customer Journey Model")
    st.markdown(f"**🔄 Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with f2:
    st.markdown("**💡 Tips:**\n- Use sidebar filters to drill down\n- Hover charts for detail")
with f3:
    st.markdown("**📧 Support:** Contact the analytics team for questions")

st.markdown("---")
st.markdown("*Sales Funnel Executive Dashboard v4.1 · Customer Journey Model · One Row Per Customer · Streamlit · Python · Plotly*")
