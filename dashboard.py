"""
Retail & RFM Dashboard — Business Overview + Customer RFM Analysis
"""
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from pathlib import Path
from urllib.request import urlopen

# Shared color palette for charts (distinct and easy to differentiate)
CHART_COLORS = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#95C623", "#6A4C93", "#1982C4", "#8AC926", "#FF595E",
    "#6A4C93", "#4ECDC4",
]

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Online Retail.xlsx"
# Portfolio downloads only (no local file fallbacks — same files as in the GitHub repo)
RETAIL_DATA_GITHUB_URL = "https://raw.githubusercontent.com/amir-elrady/amir-retail-project/main/Online%20Retail.xlsx"
RETAIL_NOTEBOOK_GITHUB_URL = (
    "https://raw.githubusercontent.com/amir-elrady/amir-retail-project/main/retail_data_analysis.ipynb"
)
RFM_NOTEBOOK_GITHUB_URL = (
    "https://raw.githubusercontent.com/amir-elrady/amir-retail-project/main/rfm_analysis.ipynb"
)
BMW_DATASET_GITHUB_URL = "https://raw.githubusercontent.com/amir-elrady/bmw-project/main/bmw_global_sales_2018_2025.csv"

# ----- Data loading & cleaning -----
@st.cache_data
def load_data_for_returns():
    """Load data for return-rate only: no dropna (keeps rows without CustomerID), no exclusion of
    credit-note invoices (C-invoices), so returns/refunds are included for an accurate return rate."""
    df = pd.read_excel(DATA_FILE)
    df = df.drop_duplicates()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    return df


@st.cache_data
def fetch_github_raw_bytes(url: str):
    """Fetch file bytes from a GitHub raw URL for st.download_button (avoids browser showing raw text)."""
    try:
        with urlopen(url, timeout=90) as response:
            return response.read()
    except Exception:
        return None

@st.cache_data
def load_and_clean():
    df = pd.read_excel(DATA_FILE)
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(subset=["CustomerID", "Description"])
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)
    df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
    df_clean["Revenue"] = df_clean["Quantity"] * df_clean["UnitPrice"]
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["InvoiceDateOnly"] = df_clean["InvoiceDate"].dt.date
    df_clean["Country"] = df_clean["Country"].replace("EIRE", "Ireland")
    return df_clean

def assign_category(desc):
    if pd.isna(desc):
        return "Other"
    d = str(desc).upper()
    if "POSTAGE" in d or desc == "POST":
        return "Postage/Fees"
    if "MANUAL" in d and len(d) < 15:
        return "Manual/Fees"
    if "T-LIGHT" in d or "T LIGHT" in d or "LANTERN" in d or "CANDLE" in d:
        return "Lighting"
    if "BAG" in d and ("JUMBO" in d or "PAPER" in d or "SHOPPING" in d):
        return "Bags"
    if "JAR" in d or "STORAGE" in d:
        return "Storage"
    if "CAKE" in d or "CAKESTAND" in d or "BOWL" in d or "MUG" in d or "TEACUP" in d:
        return "Kitchen/Dining"
    if "ORNAMENT" in d or "DECORATION" in d or "BUNTING" in d or "GARLAND" in d:
        return "Decor"
    if "CRAFT" in d or "PAPER CRAFT" in d:
        return "Craft"
    if "HOLDER" in d or "STAND" in d:
        return "Holders/Stands"
    if "HEART" in d or "HANGING" in d:
        return "Decor"
    if "TOY" in d or "GLIDER" in d or "GAME" in d:
        return "Toys/Games"
    if "SET" in d and ("PAINT" in d or "BAKING" in d):
        return "Sets"
    return "Other"

@st.cache_data
def get_rfm(df_clean):
    reference_date = df_clean["InvoiceDate"].max()
    rfm = (
        df_clean.groupby("CustomerID")
        .agg(
            last_order_date=("InvoiceDate", "max"),
            first_order_date=("InvoiceDate", "min"),
            order_count=("InvoiceNo", "nunique"),
            total_revenue=("Revenue", "sum"),
        )
        .reset_index()
    )
    rfm["Recency"] = (reference_date - rfm["last_order_date"]).dt.days
    rfm["Frequency"] = rfm["order_count"]
    rfm["Monetary"] = rfm["total_revenue"]
    rfm["R_score"] = pd.qcut(rfm["Recency"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    def rfm_segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 4 and f >= 2 and m >= 2:
            return "Loyal Customers"
        if r >= 4 and f <= 2 and m <= 2:
            return "New Customers"
        if r >= 3 and f >= 3 and m >= 3:
            return "Potential Loyalists"
        if r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        if r <= 2 and f >= 2 and m >= 2:
            return "Hibernating"
        if r <= 2 and f <= 2 and m <= 2:
            return "Lost"
        if r >= 3 and f <= 2 and m <= 2:
            return "Promising"
        return "Need Attention"

    rfm["Segment"] = rfm.apply(rfm_segment, axis=1)
    return rfm

# Segment-based recommendations
SEGMENT_RECOMMENDATIONS = {
    "Champions": "Thank them; offer exclusive early access and VIP rewards to keep them engaged.",
    "Loyal Customers": "Upsell complementary products and introduce a loyalty program tier.",
    "New Customers": "Onboard with a second-purchase incentive (e.g. free shipping) to increase frequency.",
    "Potential Loyalists": "Recommend best-sellers in categories they already buy to push them to Champions.",
    "At Risk": "Win-back campaign: personalized email with a limited-time offer on their favourite categories.",
    "Hibernating": "Re-engagement campaign with new arrivals and a small discount to bring them back.",
    "Lost": "Low-cost reactivation (e.g. one-off email); avoid expensive channels.",
    "Promising": "Encourage a second purchase with product recommendations and a small incentive.",
    "Need Attention": "Highlight popular products and limited-time offers to increase frequency and spend.",
}

# ----- Page 1: Business Overview -----
def page_business_overview(df_clean, df_returns):
    st.title("Business Overview")
    st.markdown("---")

    # KPIs (return rate from returns-inclusive data so we don't drop return transactions)
    total_revenue = df_clean["Revenue"].sum()
    n_customers = df_clean["CustomerID"].nunique()
    invoice_metrics = (
        df_clean.groupby("InvoiceNo")
        .agg(order_revenue=("Revenue", "sum"))
        .reset_index()
    )
    aov = invoice_metrics["order_revenue"].mean()
    gross_positive = df_returns[df_returns["Revenue"] > 0]["Revenue"].sum()
    returns_sum = df_returns[df_returns["Revenue"] < 0]["Revenue"].sum()
    return_rate_pct = (abs(returns_sum) / gross_positive * 100) if gross_positive > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"£{total_revenue:,.0f}")
    col2.metric("Customers", f"{n_customers:,}")
    col3.metric("AOV", f"£{aov:,.0f}")
    col4.metric("Return Rate", f"{return_rate_pct:.2f}%")
    st.markdown("---")

    # Revenue trends
    daily = (
        df_clean.groupby("InvoiceDateOnly")
        .agg(daily_revenue=("Revenue", "sum"), daily_orders=("InvoiceNo", "nunique"))
        .reset_index()
    )
    daily["InvoiceDateOnly"] = pd.to_datetime(daily["InvoiceDateOnly"])
    fig_trend = px.line(
        daily,
        x="InvoiceDateOnly",
        y="daily_revenue",
        title="Revenue Trend (Daily)",
        labels={"InvoiceDateOnly": "Date", "daily_revenue": "Revenue (£)"},
        color_discrete_sequence=[CHART_COLORS[0]],
    )
    fig_trend.update_layout(template="plotly_white", height=320)
    st.plotly_chart(fig_trend, use_container_width=True)
    st.markdown("**Observation:** Daily revenue is volatile; **Q4 is the strongest quarter** (~39% of total revenue).")
    st.markdown("---")

    # Geographic: bar + pie
    country_metrics = (
        df_clean.groupby("Country")
        .agg(
            total_revenue=("Revenue", "sum"),
            total_orders=("InvoiceNo", "nunique"),
            total_items_sold=("Quantity", "sum"),
        )
        .reset_index()
    )
    country_metrics["AOV"] = country_metrics["total_revenue"] / country_metrics["total_orders"]
    country_top = country_metrics.sort_values("total_revenue", ascending=False).head(12)

    col_geo1, col_geo2 = st.columns(2)
    with col_geo1:
        fig_geo = px.bar(
            country_top,
            x="Country",
            y="total_revenue",
            title="Revenue by Country (Top 12)",
            labels={"total_revenue": "Revenue (£)", "Country": "Country"},
            color="total_revenue",
            color_continuous_scale="Blues",
        )
        fig_geo.update_layout(template="plotly_white", height=360, xaxis_tickangle=-45, showlegend=False)
        fig_geo.update_coloraxes(showscale=False)
        st.plotly_chart(fig_geo, use_container_width=True)
    with col_geo2:
        fig_pie = px.pie(
            country_top,
            values="total_revenue",
            names="Country",
            title="Revenue Share by Country (Top 12)",
            color_discrete_sequence=CHART_COLORS,
            hole=0.35,
        )
        fig_pie.update_layout(
            template="plotly_white",
            height=500,
            margin=dict(t=60, b=40, l=20, r=140),
            legend=dict(font=dict(size=12), orientation="v", x=1.02, xanchor="left"),
            uniformtext_minsize=11,
            uniformtext_mode="hide",
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown(
        "**Observation:** UK dominates revenue; Ireland, Germany, France, Netherlands contribute smaller share. "
        "Some international markets show higher AOV (e.g. Netherlands), consistent with B2B or bulk."
    )
    st.markdown("---")

    # Product performance
    product_metrics = (
        df_clean.groupby(["StockCode", "Description"])
        .agg(total_revenue=("Revenue", "sum"), total_quantity=("Quantity", "sum"))
        .reset_index()
    )
    top_products = product_metrics.nlargest(10, "total_revenue").copy()
    top_products["Description"] = top_products["Description"].replace("Manual", "Manual (fee)")
    fig_prod = px.bar(
        top_products,
        x="Description",
        y="total_revenue",
        title="Top 10 Products by Revenue",
        labels={"total_revenue": "Revenue (£)", "Description": "Product"},
        color="total_revenue",
        color_continuous_scale="Teal",
    )
    fig_prod.update_layout(template="plotly_white", height=400, xaxis_tickangle=-45, showlegend=False)
    fig_prod.update_coloraxes(showscale=False)
    st.plotly_chart(fig_prod, use_container_width=True)
    st.markdown(
        "**Observation:** Revenue is concentrated in a small set of product categories: home decor (cakestands, T-lights, "
        "ornaments, bunting), bags, and storage. These core categories are the main levers for growth and forecasting."
    )
    st.markdown("---")

    # Actionable insights
    st.subheader("Actionable insights")
    st.markdown("- **Revenue:** Q4 drives ~39% of revenue; seasonality is material for forecasting and planning.")
    st.markdown("- **Geography:** UK is the core market; international shows higher AOV in some countries—worth segmenting for strategy.")
    st.markdown("- **Product:** Core categories (decor, kitchen, lighting, bags, storage) drive most revenue; prioritise stock and range in these and use for cross-sell and bundle opportunities.")
    st.markdown("- **Returns:** ~8–9% return rate; slice by category and country for root-cause analysis; link to RFM for win-back.")

# ----- Page 2: Customer RFM -----
def page_customer_rfm(df_clean, rfm):
    st.title("Customer RFM Analysis")
    st.markdown("---")

    # Customer selector: search + dropdown
    customer_ids = sorted(rfm["CustomerID"].astype(str).tolist())
    search = st.text_input("Search by Customer ID", placeholder="e.g. 12347")
    if search:
        options = [c for c in customer_ids if search in c]
    else:
        options = customer_ids
    choice_list = options if options else customer_ids
    selected_id = st.selectbox("Select Customer", options=choice_list, key="cust_select")
    if not selected_id:
        st.info("Select or search for a customer.")
        return
    cid = int(selected_id)
    cust_rfm = rfm[rfm["CustomerID"] == cid].iloc[0]
    segment = cust_rfm["Segment"]
    r_score = int(cust_rfm["R_score"])
    f_score = int(cust_rfm["F_score"])
    m_score = int(cust_rfm["M_score"])
    rfm_score = f"{r_score}{f_score}{m_score}"

    # Segment & recommendations
    st.subheader(f"Segment: {segment}")
    rec = SEGMENT_RECOMMENDATIONS.get(segment, "Review behavior and tailor outreach.")
    st.info(rec)
    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
    score_col1.metric("RFM Score", rfm_score)
    score_col2.metric("Recency Score (R)", r_score)
    score_col3.metric("Frequency Score (F)", f_score)
    score_col4.metric("Monetary Score (M)", m_score)
    st.markdown("---")

    # Customer profile
    st.subheader("Customer Profile")
    cust_trans = df_clean[df_clean["CustomerID"] == cid]
    country = cust_trans["Country"].mode().iloc[0] if len(cust_trans) else "—"
    first_purchase = cust_trans["InvoiceDate"].min()
    last_purchase = cust_trans["InvoiceDate"].max()
    total_rev = cust_trans["Revenue"].sum()
    n_orders = cust_trans["InvoiceNo"].nunique()
    aov_cust = total_rev / n_orders if n_orders else 0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Location**") 
        st.write(country)
        st.markdown("**First purchase**")
        st.write(first_purchase.strftime("%Y-%m-%d") if pd.notna(first_purchase) else "—")
        st.markdown("**Last purchase**")
        st.write(last_purchase.strftime("%Y-%m-%d") if pd.notna(last_purchase) else "—")
    with col2:
        st.markdown("**Total revenue**")
        st.write(f"£{total_rev:,.2f}")
        st.markdown("**AOV**")
        st.write(f"£{aov_cust:,.2f}")
        st.markdown("**Transaction count**")
        st.write(n_orders)
    st.markdown("---")

    # Top products (chart)
    st.subheader("Top Products Purchased")
    prod_cust = (
        cust_trans.groupby(["StockCode", "Description"])
        .agg(revenue=("Revenue", "sum"), quantity=("Quantity", "sum"))
        .reset_index()
    )
    prod_cust = prod_cust.nlargest(10, "revenue")
    if prod_cust.empty:
        st.caption("No product data.")
    else:
        prod_cust = prod_cust.copy()
        prod_cust["Label"] = prod_cust["Description"].apply(lambda x: (x[:35] + "…") if len(str(x)) > 35 else x)
        fig_top = px.bar(
            prod_cust,
            x="Label",
            y="revenue",
            title="",
            labels={"revenue": "Revenue (£)", "Label": "Product"},
            color="revenue",
            color_continuous_scale="Blues",
        )
        fig_top.update_layout(template="plotly_white", height=340, xaxis_tickangle=-45, showlegend=False)
        fig_top.update_coloraxes(showscale=False)
        st.plotly_chart(fig_top, use_container_width=True)

    # Categories (chart)
    st.subheader("Product Categories Purchased")
    cust_trans = cust_trans.copy()
    cust_trans["Category"] = cust_trans["Description"].apply(assign_category)
    cat_cust = (
        cust_trans.groupby("Category")
        .agg(revenue=("Revenue", "sum"))
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    if cat_cust.empty:
        st.caption("No category data.")
    else:
        fig_cat = px.pie(
            cat_cust,
            values="revenue",
            names="Category",
            title="",
            hole=0.45,
            color_discrete_sequence=CHART_COLORS,
        )
        fig_cat.update_layout(template="plotly_white", height=360)
        st.plotly_chart(fig_cat, use_container_width=True)

def page_python_code():
    st.title("Dataset and Python Code Used")
    st.markdown("---")
    st.markdown(
        "This project uses the UCI Online Retail dataset, which contains transactional records "
        "from a UK-based online gift retailer between 2010 and 2011. Each row represents an item "
        "within an invoice and includes fields such as invoice number, product description, quantity, "
        "invoice date, unit price, customer ID, and country. The analysis focuses on revenue trends, "
        "customer behavior, product performance, and RFM-based customer segmentation."
    )
    st.markdown("Download resources used in this portfolio project:")
    retail_xlsx_bytes = fetch_github_raw_bytes(RETAIL_DATA_GITHUB_URL)
    if retail_xlsx_bytes:
        st.download_button(
            "Download Dataset (Online Retail.xlsx)",
            data=retail_xlsx_bytes,
            file_name="Online Retail.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_portfolio_retail_xlsx",
        )
    else:
        st.warning("Could not load the dataset from GitHub. Refresh the page or try again shortly.")

    retail_nb_bytes = fetch_github_raw_bytes(RETAIL_NOTEBOOK_GITHUB_URL)
    if retail_nb_bytes:
        st.download_button(
            "Download Notebook (retail_data_analysis.ipynb)",
            data=retail_nb_bytes,
            file_name="retail_data_analysis.ipynb",
            mime="application/x-ipynb+json",
            key="dl_portfolio_retail_ipynb",
        )
    else:
        st.warning("Could not load retail_data_analysis.ipynb from GitHub. Refresh the page or try again shortly.")

    rfm_nb_bytes = fetch_github_raw_bytes(RFM_NOTEBOOK_GITHUB_URL)
    if rfm_nb_bytes:
        st.download_button(
            "Download Notebook (rfm_analysis.ipynb)",
            data=rfm_nb_bytes,
            file_name="rfm_analysis.ipynb",
            mime="application/x-ipynb+json",
            key="dl_portfolio_rfm_ipynb",
        )
    else:
        st.warning("Could not load rfm_analysis.ipynb from GitHub. Refresh the page or try again shortly.")
    st.markdown("Key code used to build this analysis dashboard:")
    st.markdown("---")

    code_1 = """import pandas as pd

df = pd.read_excel("Online Retail.xlsx")
df_clean = df.copy()
df_clean = df_clean.drop_duplicates()
df_clean = df_clean.dropna(subset=["CustomerID", "Description"])
df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)
df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
df_clean["Revenue"] = df_clean["Quantity"] * df_clean["UnitPrice"]
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
df_clean["InvoiceDateOnly"] = df_clean["InvoiceDate"].dt.date
df_clean["Country"] = df_clean["Country"].replace("EIRE", "Ireland")
"""
    with st.expander("1) Data loading and cleaning", expanded=True):
        st.code(code_1, language="python")
        st.download_button(
            label="Download this section",
            data=code_1,
            file_name="01_data_cleaning.py",
            mime="text/x-python",
            key="download_code_1",
        )

    code_2 = """df_returns = pd.read_excel("Online Retail.xlsx").drop_duplicates()
df_returns["Revenue"] = df_returns["Quantity"] * df_returns["UnitPrice"]
gross_positive = df_returns[df_returns["Revenue"] > 0]["Revenue"].sum()
returns_sum = df_returns[df_returns["Revenue"] < 0]["Revenue"].sum()
return_rate_pct = (abs(returns_sum) / gross_positive * 100) if gross_positive > 0 else 0
"""
    with st.expander("2) Return-rate calculation (includes credit notes)"):
        st.code(code_2, language="python")
        st.download_button(
            label="Download this section",
            data=code_2,
            file_name="02_return_rate.py",
            mime="text/x-python",
            key="download_code_2",
        )

    code_3 = """total_revenue = df_clean["Revenue"].sum()
n_customers = df_clean["CustomerID"].nunique()
invoice_metrics = df_clean.groupby("InvoiceNo").agg(order_revenue=("Revenue", "sum")).reset_index()
aov = invoice_metrics["order_revenue"].mean()

daily = df_clean.groupby("InvoiceDateOnly").agg(
    daily_revenue=("Revenue", "sum"),
    daily_orders=("InvoiceNo", "nunique")
).reset_index()

country_metrics = df_clean.groupby("Country").agg(
    total_revenue=("Revenue", "sum"),
    total_orders=("InvoiceNo", "nunique"),
    total_items_sold=("Quantity", "sum")
).reset_index()
country_metrics["AOV"] = country_metrics["total_revenue"] / country_metrics["total_orders"]
"""
    with st.expander("3) Business overview metrics"):
        st.code(code_3, language="python")
        st.download_button(
            label="Download this section",
            data=code_3,
            file_name="03_business_overview_metrics.py",
            mime="text/x-python",
            key="download_code_3",
        )

    code_4 = """reference_date = df_clean["InvoiceDate"].max()
rfm = df_clean.groupby("CustomerID").agg(
    last_order_date=("InvoiceDate", "max"),
    order_count=("InvoiceNo", "nunique"),
    total_revenue=("Revenue", "sum")
).reset_index()

rfm["Recency"] = (reference_date - rfm["last_order_date"]).dt.days
rfm["Frequency"] = rfm["order_count"]
rfm["Monetary"] = rfm["total_revenue"]

rfm["R_score"] = pd.qcut(rfm["Recency"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
"""
    with st.expander("4) RFM analysis"):
        st.code(code_4, language="python")
        st.download_button(
            label="Download this section",
            data=code_4,
            file_name="04_rfm_analysis.py",
            mime="text/x-python",
            key="download_code_4",
        )

def page_bmw_sql_queries():
    st.title("SQL Queries Used")
    st.markdown("---")
    st.caption("Dataset is synthetic, designed to simulate BMW global sales patterns for the purpose of demonstrating SQL and Tableau skills.")
    st.markdown(
        "This project uses a synthetic BMW global sales dataset covering monthly performance from "
        "2018 to 2025 across four regions (Europe, China, USA, and Rest of World). The data includes "
        "model-level units sold, average price, revenue, BEV adoption share, premium share, GDP growth, "
        "and fuel price index. The SQL analysis highlights data cleaning, EV adoption trends, revenue mix "
        "by model category, and macroeconomic sensitivity."
    )
    if BMW_DATASET_GITHUB_URL:
        remote_csv = fetch_github_raw_bytes(BMW_DATASET_GITHUB_URL)
        if remote_csv:
            st.download_button(
                "Download BMW Dataset (CSV)",
                data=remote_csv,
                file_name="bmw_global_sales_2018_2025.csv",
                mime="text/csv",
                key="dl_portfolio_bmw_csv",
            )
        else:
            st.warning("Could not load the BMW CSV from GitHub. Refresh the page or try again shortly.")
    st.markdown("SQL queries used for the BMW Global Sales Project:")
    st.markdown("---")

    query_1 = """-- Data Cleaning & Transformation
-- Source table:  bmw_global_sales  (raw CSV upload)
-- Output table:  bmw_global_sales_clean
--
-- Pre-cleaning audit results:
--   Rows: 3,072 — matches expected count (8 years x 4 regions x 12 months x 8 models)
--   Nulls: 0 across all columns
--   Duplicates: 0
--   Revenue: Verified as Units_Sold x Avg_Price_EUR with no discrepancies
--
-- Issues found:
--   9 rows contain slightly negative BEV_Share values (e.g. -0.001, -0.015),
--   all from 2018. These are floating point rounding errors, not real negatives.
--   Fixed by clamping to 0 using GREATEST(BEV_Share, 0).
--
-- Enhancements added:
--   - Year_Month: proper DATE column for time-series work
--   - Model_Category: SUV / Sedan / Electric / MINI grouping for segmentation
--   - BEV_Share_Pct: BEV_Share converted to percentage (x100) for readability
--   - data_quality_flag: marks the 9 corrected rows with original value

CREATE OR REPLACE TABLE `bmw-data-491218.bmw_sales.bmw_global_sales_clean` AS
WITH source AS (SELECT * FROM `bmw-data-491218.bmw_sales.bmw_global_sales`)
SELECT
  Year, Month,
  DATE(Year, Month, 1) AS Year_Month,
  Region, Model,
  CASE
    WHEN Model IN ('X3','X5','X7') THEN 'SUV'
    WHEN Model IN ('3 Series','5 Series') THEN 'Sedan'
    WHEN Model IN ('i4','iX') THEN 'Electric'
    ELSE 'MINI'
  END AS Model_Category,
  Units_Sold, Avg_Price_EUR, Revenue_EUR,
  GREATEST(BEV_Share, 0) AS BEV_Share,
  ROUND(GREATEST(BEV_Share, 0) * 100, 2) AS BEV_Share_Pct,
  Premium_Share, GDP_Growth, Fuel_Price_Index,
  CASE WHEN BEV_Share < 0
    THEN 'BEV_Share corrected (was ' || CAST(ROUND(BEV_Share,4) AS STRING) || ')'
    ELSE 'OK'
  END AS data_quality_flag
FROM source;

-- Verification queries:
-- SELECT COUNT(*) AS total_rows FROM `bmw-data-491218.bmw_sales.bmw_global_sales_clean`;
-- SELECT COUNT(*) AS neg_bev_remaining FROM `bmw-data-491218.bmw_sales.bmw_global_sales_clean` WHERE BEV_Share < 0;
-- SELECT * FROM `bmw-data-491218.bmw_sales.bmw_global_sales_clean` WHERE data_quality_flag != 'OK';"""
    with st.expander("1) Cleaning the data", expanded=True):
        st.code(query_1, language="sql")
        st.download_button(
            label="Download this query",
            data=query_1,
            file_name="01_data_cleaning_transformation.sql",
            mime="text/sql",
            key="download_sql_1",
        )

    query_2 = """-- Q1: EV Adoption Trajectory by Region (2018–2025)
--
-- Question: Which regions are leading BMW's shift to electric, and how quickly
-- is that share growing year over year?
--
-- Approach: Aggregate BEV_Share_Pct to annual averages per region, then apply
-- a LAG window function to calculate YoY growth rate. BEV_Share_Pct is already
-- cleaned and pre-computed in the source table.
--
-- Note: YoY growth returns NULL for 2018 (no prior year to compare against).
-- This is expected and correct — do not treat it as missing data.

CREATE OR REPLACE TABLE `bmw-data-491218.bmw_sales.bmw_ev_adoption_by_region` AS
WITH annual_bev AS (
  SELECT
    Year, Region,
    ROUND(AVG(BEV_Share_Pct), 2) AS Avg_BEV_Share_Pct,
    SUM(Units_Sold) AS Total_Units_Sold,
    ROUND(SUM(Revenue_EUR) / 1e9, 2) AS Total_Revenue_BEUR
  FROM `bmw-data-491218.bmw_sales.bmw_global_sales_clean`
  GROUP BY Year, Region
)
SELECT
  Year, Region, Avg_BEV_Share_Pct, Total_Units_Sold, Total_Revenue_BEUR,
  ROUND(
    (Avg_BEV_Share_Pct - LAG(Avg_BEV_Share_Pct) OVER (PARTITION BY Region ORDER BY Year))
    / NULLIF(LAG(Avg_BEV_Share_Pct) OVER (PARTITION BY Region ORDER BY Year), 0) * 100, 1
  ) AS YoY_BEV_Growth_Pct
FROM annual_bev
ORDER BY Region, Year;"""
    with st.expander("2) EV Adoption Trajectory"):
        st.code(query_2, language="sql")
        st.download_button(
            label="Download this query",
            data=query_2,
            file_name="02_ev_adoption_trajectory.sql",
            mime="text/sql",
            key="download_sql_2",
        )

    query_3 = """-- Q2: Revenue Mix by Model Category Over Time (2018–2025)
--
-- Question: Which model category drives BMW's revenue, and is the product mix
-- shifting meaningfully toward electric vehicles?
--
-- Approach: Group by Year and Model_Category (derived during cleaning) to get
-- annual revenue, unit volume, and average price per segment. A window function
-- calculates each segment's share of total revenue within a given year.
--
-- Model categories: SUV (X3/X5/X7), Sedan (3 Series/5 Series),
--                   Electric (i4/iX), MINI

CREATE OR REPLACE TABLE `bmw-data-491218.bmw_sales.bmw_revenue_mix_by_category` AS
WITH segment_totals AS (
  SELECT
    Year, Model_Category,
    ROUND(SUM(Revenue_EUR) / 1e9, 2) AS Revenue_BEUR,
    SUM(Units_Sold) AS Total_Units,
    ROUND(AVG(Avg_Price_EUR), 0) AS Avg_Model_Price_EUR
  FROM `bmw-data-491218.bmw_sales.bmw_global_sales_clean`
  GROUP BY Year, Model_Category
)
SELECT
  Year, Model_Category, Revenue_BEUR, Total_Units, Avg_Model_Price_EUR,
  ROUND(Revenue_BEUR * 100.0 / SUM(Revenue_BEUR) OVER (PARTITION BY Year), 1) AS Revenue_Share_Pct
FROM segment_totals
ORDER BY Year, Revenue_BEUR DESC;"""
    with st.expander("3) Revenue Comparison"):
        st.code(query_3, language="sql")
        st.download_button(
            label="Download this query",
            data=query_3,
            file_name="03_revenue_comparison.sql",
            mime="text/sql",
            key="download_sql_3",
        )

    query_4 = """-- Q3: Macro Sensitivity — GDP Growth & Fuel Price vs. Sales Volume (2018–2025)
--
-- Question: How sensitive is BMW's demand to macroeconomic conditions,
-- and which region shows the strongest relationship between economic factors
-- and sales performance?
--
-- Approach: Aggregate to annual totals per region, then calculate YoY unit
-- growth alongside the prevailing GDP growth and fuel price environment.
-- The relationship between macro indicators and sales volume is weak across
-- all regions, but this output lets you visually inspect specific
-- year-over-year dynamics and flag any outlier periods worth investigating.
--
-- Note: GDP_Growth and Fuel_Price_Index are region-level macro indicators
-- averaged across all models for a given region-year. YoY growth is NULL
-- for 2018 (base year).

CREATE OR REPLACE TABLE `bmw-data-491218.bmw_sales.bmw_macro_vs_sales` AS
WITH regional_annual AS (
  SELECT
    Region, Year,
    SUM(Units_Sold) AS Total_Units,
    ROUND(AVG(GDP_Growth), 3) AS Avg_GDP_Growth,
    ROUND(AVG(Fuel_Price_Index), 3) AS Avg_Fuel_Price_Idx,
    ROUND(SUM(Revenue_EUR) / 1e9, 2) AS Total_Revenue_BEUR
  FROM `bmw-data-491218.bmw_sales.bmw_global_sales_clean`
  GROUP BY Region, Year
)
SELECT
  Region, Year, Total_Units, Avg_GDP_Growth, Avg_Fuel_Price_Idx, Total_Revenue_BEUR,
  ROUND(
    (Total_Units - LAG(Total_Units) OVER (PARTITION BY Region ORDER BY Year)) * 100.0
    / NULLIF(LAG(Total_Units) OVER (PARTITION BY Region ORDER BY Year), 0), 1
  ) AS Units_YoY_Growth_Pct
FROM regional_annual
ORDER BY Region, Year;"""
    with st.expander("4) GDP & Fuel vs Sales"):
        st.code(query_4, language="sql")
        st.download_button(
            label="Download this query",
            data=query_4,
            file_name="04_gdp_fuel_vs_sales.sql",
            mime="text/sql",
            key="download_sql_4",
        )

def page_bmw_tableau_visualizations():
    st.title("Tableau Visualizations")
    st.markdown("---")
    st.caption("Dataset is synthetic, designed to simulate BMW global sales patterns for the purpose of demonstrating SQL and Tableau skills.")
    st.warning("For the best viewing experience and correct layout, please switch to full-screen mode when exploring this dashboard.")
    tableau_url = "https://public.tableau.com/views/bmwsalesviz/Dashboard1?:showVizHome=no"
    st.link_button("Open Tableau in new tab", tableau_url)
    components.iframe(tableau_url, height=900, scrolling=True)

# ----- Main -----
def main():
    st.set_page_config(page_title="Amir's Portfolio", page_icon="📊", layout="wide")
    st.sidebar.title("Amir's Portfolio")
    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("### Select Portfolio Project")
    project = st.sidebar.selectbox(
        "Select Portfolio Project",
        ["Select a project...", "Online Retail Project", "BMW Global Sales Analysis"],
        label_visibility="collapsed",
    )

    if project == "Select a project...":
        st.markdown("### Welcome to my portfolio")
        st.info("Please select a project from the left sidebar to begin.")
    elif project == "Online Retail Project":
        st.sidebar.caption("Data: Online Retail")
        page = st.sidebar.radio("Analysis", ["Dataset and Python Code Used", "Customer Overview", "RFM Analysis"])

        df_clean = load_and_clean()
        rfm = get_rfm(df_clean)
        df_returns = load_data_for_returns()

        if page == "Dataset and Python Code Used":
            page_python_code()
        elif page == "Customer Overview":
            page_business_overview(df_clean, df_returns)
        else:
            page_customer_rfm(df_clean, rfm)
    else:
        st.sidebar.caption("Data: BMW Global Sales")
        bmw_page = st.sidebar.radio("Analysis", ["SQL Queries Used", "Tableau Visualizations"])
        if bmw_page == "SQL Queries Used":
            page_bmw_sql_queries()
        else:
            page_bmw_tableau_visualizations()

if __name__ == "__main__":
    main()
