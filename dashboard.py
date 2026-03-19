"""
Retail & RFM Dashboard — Business Overview + Customer RFM Analysis
"""
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Shared color palette for charts (distinct and easy to differentiate)
CHART_COLORS = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#95C623", "#6A4C93", "#1982C4", "#8AC926", "#FF595E",
    "#6A4C93", "#4ECDC4",
]

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Online Retail.xlsx"
RETAIL_NOTEBOOK = BASE_DIR / "retail_data_analysis.ipynb"
RFM_NOTEBOOK = BASE_DIR / "rfm_analysis.ipynb"

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

    # Segment & recommendations
    st.subheader(f"Segment: {segment}")
    rec = SEGMENT_RECOMMENDATIONS.get(segment, "Review behavior and tailor outreach.")
    st.info(rec)
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
    st.title("Python Code")
    st.markdown("---")
    st.markdown("Key code used to build this analysis dashboard.")
    st.markdown("Download resources used in this portfolio project:")
    with open(DATA_FILE, "rb") as f:
        st.download_button(
            label="Download Dataset (Online Retail.xlsx)",
            data=f.read(),
            file_name="Online Retail.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with open(RETAIL_NOTEBOOK, "rb") as f:
        st.download_button(
            label="Download Notebook (retail_data_analysis.ipynb)",
            data=f.read(),
            file_name="retail_data_analysis.ipynb",
            mime="application/x-ipynb+json",
        )
    with open(RFM_NOTEBOOK, "rb") as f:
        st.download_button(
            label="Download Notebook (rfm_analysis.ipynb)",
            data=f.read(),
            file_name="rfm_analysis.ipynb",
            mime="application/x-ipynb+json",
        )
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

# ----- Main -----
def main():
    st.set_page_config(page_title="Retail & RFM Dashboard", page_icon="📊", layout="wide")
    st.sidebar.title("Navigation")
    project = st.sidebar.selectbox("Project", ["Retail & RFM Analysis"])
    st.sidebar.caption("More projects can be added here.")

    if project == "Retail & RFM Analysis":
        st.sidebar.caption("Data: Online Retail (cleaned)")
        page = st.sidebar.radio("Analysis", ["Customer Overview", "RFM Analysis", "Python Code"])

        df_clean = load_and_clean()
        rfm = get_rfm(df_clean)
        df_returns = load_data_for_returns()

        if page == "Customer Overview":
            page_business_overview(df_clean, df_returns)
        elif page == "RFM Analysis":
            page_customer_rfm(df_clean, rfm)
        else:
            page_python_code()

if __name__ == "__main__":
    main()
