"""
============================================================
  Property Value Estimator
  Built with Python, Streamlit, and scikit-learn
  Author: Your Name
  Description: A professional ML web app that estimates
               property values using Linear Regression.
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import io
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Property Value Estimator",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  CUSTOM CSS — Clean, dark-accent design
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ══ GLOBAL RESET ══ */
html, body { background: #0e0b1a !important; }

.stApp { background: #0e0b1a !important; }

/* Kill ALL blue highlights Streamlit adds */
* { font-family: 'Inter', sans-serif !important; }

/* ══ FORCE ALL TEXT DARK-THEME WHITE ══ */
p, span, div, li, td, th, label, small, strong, b, i, em,
h1, h2, h3, h4, h5, h6,
[class*="st-"], [data-testid] {
    color: #ede8f5 !important;
    background-color: transparent !important;
}

/* ══ MAIN AREA ══ */
section.main, section.main > div, .block-container {
    background: #0e0b1a !important;
    padding-top: 1.5rem !important;
}

/* ══ SIDEBAR ══ */
section[data-testid="stSidebar"] {
    background: #0a0814 !important;
    border-right: 1px solid #2d1f4e !important;
}
section[data-testid="stSidebar"] > div {
    background: #0a0814 !important;
}
section[data-testid="stSidebar"] * {
    color: #cfc8e8 !important;
    background: transparent !important;
}

/* Sidebar radio - remove blue pill highlight */
section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    background: transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #cfc8e8 !important;
    background: transparent !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #1e1535 !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #cfc8e8 !important;
    background: transparent !important;
}

/* ══ WIDGETS - remove blue background on labels ══ */
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: #b8a8d8 !important;
    background: transparent !important;
    font-weight: 500 !important;
}

/* Slider values */
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSlider"] p {
    color: #8a7aaa !important;
    background: transparent !important;
}

/* Number input box */
[data-testid="stNumberInput"] input {
    background: #1a1528 !important;
    color: #ede8f5 !important;
    border: 1px solid #3d2f62 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stNumberInput"] button {
    background: #2a1f45 !important;
    color: #ede8f5 !important;
    border-color: #3d2f62 !important;
}

/* ══ STREAMLIT METRIC WIDGET ══ */
[data-testid="metric-container"] {
    background: #1a1528 !important;
    border: 1px solid #2d1f4e !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] p { color: #8a7aaa !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"]   { color: #a78bfa !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"]   { color: #34d399 !important; }

/* ══ BUTTONS ══ */
.stButton > button {
    background: linear-gradient(135deg, #2d1b69, #6d28d9) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stButton > button * { color: #ffffff !important; background: transparent !important; }

.stDownloadButton > button {
    background: #1a1528 !important;
    color: #a78bfa !important;
    border: 2px solid #6d28d9 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100% !important;
}
.stDownloadButton > button * { color: #a78bfa !important; background: transparent !important; }

/* ══ DATAFRAME / TABLE ══ */
[data-testid="stDataFrame"] iframe { background: #1a1528 !important; }
.stDataFrame { background: #1a1528 !important; border-radius: 10px !important; }

/* ══ MARKDOWN TEXT ══ */
.stMarkdown p, .stMarkdown li { color: #cfc8e8 !important; }
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #ffffff !important; }
.stMarkdown strong, .stMarkdown b { color: #c4b5fd !important; }
.stMarkdown code {
    background: #1e1535 !important;
    color: #86efac !important;
    padding: 2px 6px; border-radius: 4px;
}

/* Code blocks */
[data-testid="stCode"], .stCodeBlock {
    background: #1a1528 !important;
}
[data-testid="stCode"] * { color: #86efac !important; }

/* ══ CUSTOM HTML COMPONENTS ══ */

/* Hero */
.hero-banner {
    background: linear-gradient(135deg, #1e0b3e 0%, #3b1a78 50%, #6d28d9 100%);
    border-radius: 18px; padding: 2.4rem 2.8rem;
    margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero-banner h1 { color: #ffffff !important; font-size: 2.2rem; font-weight: 700; margin: 0 0 0.4rem 0; }
.hero-banner p  { color: #c4b5fd !important; font-size: 1rem; margin: 0; }
.hero-tag {
    display: inline-block; background: rgba(255,255,255,0.15);
    color: #e9d5ff !important; font-size: 0.75rem; font-weight: 600;
    padding: 4px 12px; border-radius: 20px; margin-bottom: 0.9rem;
    letter-spacing: 1px; text-transform: uppercase;
}

/* Section title */
.section-title {
    font-size: 1.15rem; font-weight: 700; color: #ffffff !important;
    margin: 0.2rem 0 1rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid #2d1f4e;
    display: flex; align-items: center; gap: 0.5rem;
}

/* Metric cards (custom HTML) */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1; background: #1a1528 !important;
    border-radius: 14px; padding: 1.4rem 1.6rem;
    border: 1px solid #2d1f4e;
    box-shadow: 0 2px 12px rgba(109,40,217,0.2); text-align: center;
}
.metric-card .metric-value {
    font-size: 1.9rem; font-weight: 700; color: #a78bfa !important;
    font-family: 'JetBrains Mono', monospace !important; line-height: 1;
}
.metric-card .metric-label {
    font-size: 0.78rem; color: #8a7aaa !important;
    font-weight: 600; margin-top: 0.4rem;
    text-transform: uppercase; letter-spacing: 0.8px;
}
.metric-card .metric-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }

/* Prediction box */
.prediction-box {
    background: linear-gradient(135deg, #1e0b3e, #6d28d9);
    border-radius: 16px; padding: 2rem 2.4rem;
    text-align: center; margin-top: 1rem;
    box-shadow: 0 8px 24px rgba(109,40,217,0.4);
}
.prediction-box .pred-label { color: #c4b5fd !important; font-size: 0.85rem; font-weight: 600; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 0.5rem; }
.prediction-box .pred-value { color: #ffffff !important; font-size: 2.6rem; font-weight: 700; font-family: 'JetBrains Mono', monospace !important; }
.prediction-box .pred-sub   { color: #ddd6fe !important; font-size: 0.82rem; margin-top: 0.5rem; }

/* Tip box */
.tip-box {
    background: #1a0f33; border-left: 4px solid #6d28d9;
    border-radius: 8px; padding: 0.9rem 1.2rem;
    margin: 1rem 0; font-size: 0.88rem; color: #cfc8e8 !important;
}
.tip-box b { color: #c4b5fd !important; }

/* Card */
.card {
    background: #1a1528 !important; border-radius: 14px;
    padding: 1.6rem; border: 1px solid #2d1f4e;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3); margin-bottom: 1.2rem;
    color: #cfc8e8 !important;
}
.card b, .card strong { color: #c4b5fd !important; font-weight: 700; }
.card code {
    background: #1e1535 !important; color: #86efac !important;
    padding: 2px 7px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem;
}

/* Footer */
.footer {
    text-align: center; color: #5b4a80 !important;
    font-size: 0.78rem; margin-top: 2.5rem;
    padding: 1.2rem 0; border-top: 1px solid #2d1f4e;
}

/* Placeholder box */
div[style*="background:#f0f4f8"] {
    background: #1a1528 !important;
    border: 1px solid #2d1f4e !important;
    border-radius: 14px !important;
}

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  HELPER FUNCTIONS
# ──────────────────────────────────────────────

MODEL_PATH = "house_price_model.pkl"

@st.cache_data
def load_data():
    """Load the house price dataset from CSV."""
    df = pd.read_csv("house_data.csv")
    return df


def train_model(df):
    """
    Train a Linear Regression model on the dataset.
    Returns the trained model + evaluation metrics.
    """
    # Features (X) and Target (y)
    X = df[["Square_Feet", "Bedrooms", "Bathrooms"]]
    y = df["Price"]

    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions on test set
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Save model to disk
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model, r2, mae, rmse, X_test, y_test, y_pred


def load_saved_model():
    """Load model from disk if it exists."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def predict_price(model, sqft, bedrooms, bathrooms):
    """Predict house price given input features."""
    input_df = pd.DataFrame({
        "Square_Feet": [sqft],
        "Bedrooms":    [bedrooms],
        "Bathrooms":   [bathrooms],
    })
    price = model.predict(input_df)[0]
    return max(price, 0)  # No negative prices


# ──────────────────────────────────────────────
#  VISUALIZATION FUNCTIONS
# ──────────────────────────────────────────────

def set_plot_style():
    """Apply a clean, consistent style to all plots."""
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.facecolor":    "#fafcfe",
        "figure.facecolor":  "#ffffff",
    })


def plot_actual_vs_predicted(y_test, y_pred):
    """Scatter plot of actual vs predicted prices."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.scatter(y_test, y_pred, color="#2c5364", alpha=0.7, edgecolors="#0f2027", s=60)

    # Perfect prediction line
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect Prediction")

    ax.set_xlabel("Actual Price ($)", fontsize=10, fontweight="600")
    ax.set_ylabel("Predicted Price ($)", fontsize=10, fontweight="600")
    ax.set_title("Actual vs Predicted Prices", fontsize=12, fontweight="700", pad=12)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    plt.tight_layout()
    return fig


def plot_residuals(y_test, y_pred):
    """Residual plot to check model errors."""
    set_plot_style()
    residuals = np.array(y_test) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.scatter(y_pred, residuals, color="#203a43", alpha=0.65, edgecolors="#0f2027", s=55)
    ax.axhline(0, color="#e74c3c", linestyle="--", linewidth=1.5)

    ax.set_xlabel("Predicted Price ($)", fontsize=10, fontweight="600")
    ax.set_ylabel("Residuals ($)", fontsize=10, fontweight="600")
    ax.set_title("Residual Plot", fontsize=12, fontweight="700", pad=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    plt.tight_layout()
    return fig


def plot_feature_correlation(df):
    """Heatmap of feature correlations."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle only
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10, "weight": "600"},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=12, fontweight="700", pad=12)
    plt.tight_layout()
    return fig


def plot_price_distribution(df):
    """Histogram of house price distribution."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.hist(df["Price"], bins=20, color="#2c5364", edgecolor="#0f2027", alpha=0.85)
    ax.axvline(df["Price"].mean(), color="#e74c3c", linestyle="--", linewidth=1.8,
               label=f'Mean: ${df["Price"].mean():,.0f}')

    ax.set_xlabel("Price ($)", fontsize=10, fontweight="600")
    ax.set_ylabel("Frequency", fontsize=10, fontweight="600")
    ax.set_title("Price Distribution", fontsize=12, fontweight="700", pad=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_sqft_vs_price(df):
    """Scatter plot of Square Feet vs Price."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))

    scatter = ax.scatter(
        df["Square_Feet"], df["Price"],
        c=df["Bedrooms"], cmap="Blues",
        edgecolors="#0f2027", alpha=0.75, s=55
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Bedrooms", fontsize=9)

    ax.set_xlabel("Square Feet", fontsize=10, fontweight="600")
    ax.set_ylabel("Price ($)", fontsize=10, fontweight="600")
    ax.set_title("Square Feet vs Price", fontsize=12, fontweight="700", pad=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    plt.tight_layout()
    return fig


def plot_feature_importance(model):
    """Bar chart of Linear Regression coefficients as feature importance."""
    set_plot_style()
    features = ["Square Feet", "Bedrooms", "Bathrooms"]
    coefs    = model.coef_

    colors = ["#0f2027", "#203a43", "#2c5364"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(features, coefs, color=colors, edgecolor="#0a1a22", height=0.5)

    for bar, val in zip(bars, coefs):
        ax.text(
            val + max(coefs)*0.01, bar.get_y() + bar.get_height()/2,
            f"${val:,.0f}", va="center", fontsize=9, fontweight="600", color="#1a2332"
        )

    ax.set_xlabel("Coefficient ($ per unit increase)", fontsize=10, fontweight="600")
    ax.set_title("Feature Coefficients", fontsize=12, fontweight="700", pad=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="sidebar-title">🏘️ Property Value Estimator</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**About this App**")
    st.markdown(
        "An AI-powered tool to estimate residential property values "
        "using **Linear Regression** trained on real estate data."
    )
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- Algorithm: Linear Regression")
    st.markdown("- Library: scikit-learn")
    st.markdown("- Features: Square Feet, Bedrooms, Bathrooms")
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.markdown("🐍 Python 3.10+  \n📊 Streamlit  \n🤖 scikit-learn  \n📈 Matplotlib / Seaborn")
    st.markdown("---")
    st.markdown("**Navigation**")
    page = st.radio(
        "Go to",
        ["🏠 Predict", "📊 Dataset & Metrics", "📈 Visualizations", "ℹ️ How It Works"],
        label_visibility="collapsed"
    )


# ──────────────────────────────────────────────
#  LOAD DATA & TRAIN MODEL (cached)
# ──────────────────────────────────────────────

df = load_data()

# Train every session (fast enough on this dataset size)
model, r2, mae, rmse, X_test, y_test, y_pred = train_model(df)


# ──────────────────────────────────────────────
#  HERO BANNER
# ──────────────────────────────────────────────

st.markdown("""
<div class="hero-banner">
    <div class="hero-tag">🤖 Machine Learning · Linear Regression</div>
    <h1>🏘️ Property Value Estimator</h1>
    <p>Enter your property details and receive an instant AI-powered valuation — powered by Python, scikit-learn & Streamlit.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 1 — PREDICT
# ══════════════════════════════════════════════

if page == "🏠 Predict":

    st.markdown('<p class="section-title">🔢 Enter Property Details</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        sqft = st.slider(
            "📐 Square Feet",
            min_value=500,
            max_value=5000,
            value=1800,
            step=50,
            help="Total living area of the house in square feet"
        )

        c1, c2 = st.columns(2)
        with c1:
            bedrooms = st.number_input(
                "🛏️ Bedrooms",
                min_value=1, max_value=10,
                value=3, step=1,
                help="Number of bedrooms"
            )
        with c2:
            bathrooms = st.number_input(
                "🚿 Bathrooms",
                min_value=1, max_value=8,
                value=2, step=1,
                help="Number of bathrooms"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Quick summary
        st.markdown(f"""
        <div class="tip-box">
            📌 <b>Property Summary:</b> &nbsp;
            {sqft:,} sq ft &nbsp;·&nbsp; {int(bedrooms)} bed &nbsp;·&nbsp; {int(bathrooms)} bath
        </div>
        """, unsafe_allow_html=True)

        predict_btn = st.button("💎 Estimate Property Value")

    with col2:
        st.markdown('<p class="section-title">📊 Model Performance</p>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">{r2:.3f}</div>
                <div class="metric-label">R² Score</div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-icon">📉</div>
                <div class="metric-value">${mae:,.0f}</div>
                <div class="metric-label">MAE</div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">📐</div>
                <div class="metric-value">${rmse:,.0f}</div>
                <div class="metric-label">RMSE</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show prediction result when button is clicked
        if predict_btn:
            price = predict_price(model, sqft, bedrooms, bathrooms)
            st.markdown(f"""
            <div class="prediction-box">
                <div class="pred-label">Estimated Property Value</div>
                <div class="pred-value">${price:,.0f}</div>
                <div class="pred-sub">Based on {sqft:,} sq ft · {int(bedrooms)} bed · {int(bathrooms)} bath</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1a1528;border:1px solid #2d1f4e;border-radius:14px;padding:2rem;text-align:center;">
                <div style="font-size:2.5rem">🏘️</div>
                <div style="font-weight:600;margin-top:0.5rem;color:#cfc8e8;">Fill in details and click Estimate</div>
                <div style="font-size:0.83rem;margin-top:0.3rem;color:#8a7aaa;">Your property valuation will appear here</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  PAGE 2 — DATASET & METRICS
# ══════════════════════════════════════════════

elif page == "📊 Dataset & Metrics":

    st.markdown('<p class="section-title">📋 Dataset Preview</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Features", 3)
    col3.metric("Avg Price", f"${df['Price'].mean():,.0f}")
    col4.metric("Avg Sq Ft", f"{df['Square_Feet'].mean():,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df.style.format({
        "Price":       "${:,.0f}",
        "Square_Feet": "{:,}",
    }), use_container_width=True, height=300)

    # Download button
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Dataset (CSV)",
        data=csv_buffer.getvalue(),
        file_name="house_data.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown('<p class="section-title">🧮 Descriptive Statistics</p>', unsafe_allow_html=True)
    st.dataframe(df.describe().style.format("{:.2f}"), use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🤖 Model Evaluation Metrics</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">{r2:.4f}</div>
            <div class="metric-label">R² Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📉</div>
            <div class="metric-value">${mae:,.0f}</div>
            <div class="metric-label">Mean Absolute Error</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📐</div>
            <div class="metric-value">${rmse:,.0f}</div>
            <div class="metric-label">Root Mean Sq. Error</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="tip-box">
        ✅ <b>R² = {r2:.4f}</b> — The model explains <b>{r2*100:.1f}%</b> of the variance in house prices. 
        An R² closer to 1.0 indicates a better fit. MAE of <b>${mae:,.0f}</b> means predictions are 
        off by roughly that amount on average.
    </div>
    """, unsafe_allow_html=True)

    # Model coefficients
    st.markdown("---")
    st.markdown('<p class="section-title">📌 Model Coefficients</p>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({
        "Feature":     ["Square Feet", "Bedrooms", "Bathrooms"],
        "Coefficient": model.coef_,
        "Interpretation": [
            f"Each extra sq ft adds ~${model.coef_[0]:,.2f} to price",
            f"Each extra bedroom adds ~${model.coef_[1]:,.2f} to price",
            f"Each extra bathroom adds ~${model.coef_[2]:,.2f} to price",
        ]
    })
    st.dataframe(coef_df.style.format({"Coefficient": "${:,.2f}"}), use_container_width=True)
    st.markdown(f"**Intercept (Base Price):** `${model.intercept_:,.2f}`")


# ══════════════════════════════════════════════
#  PAGE 3 — VISUALIZATIONS
# ══════════════════════════════════════════════

elif page == "📈 Visualizations":

    st.markdown('<p class="section-title">📈 Data & Model Visualizations</p>', unsafe_allow_html=True)

    # Row 1
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("**Actual vs Predicted Prices**")
        st.pyplot(plot_actual_vs_predicted(y_test, y_pred))
    with c2:
        st.markdown("**Residual Plot**")
        st.pyplot(plot_residuals(y_test, y_pred))

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        st.markdown("**Feature Correlation Heatmap**")
        st.pyplot(plot_feature_correlation(df))
    with c4:
        st.markdown("**Price Distribution**")
        st.pyplot(plot_price_distribution(df))

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 3
    c5, c6 = st.columns(2, gap="medium")
    with c5:
        st.markdown("**Square Feet vs Price**")
        st.pyplot(plot_sqft_vs_price(df))
    with c6:
        st.markdown("**Feature Coefficients (Importance)**")
        st.pyplot(plot_feature_importance(model))


# ══════════════════════════════════════════════
#  PAGE 4 — HOW IT WORKS
# ══════════════════════════════════════════════

elif page == "ℹ️ How It Works":

    st.markdown('<p class="section-title">ℹ️ How the ML Model Works</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <b>1. Data Collection</b><br>
        A CSV dataset of 100 house records is loaded, each with Square Feet, Bedrooms, Bathrooms, and Price.
    </div>
    <div class="card">
        <b>2. Feature Engineering</b><br>
        Three input features are selected: <code>Square_Feet</code>, <code>Bedrooms</code>, and <code>Bathrooms</code>.
        These are the variables the model uses to make predictions.
    </div>
    <div class="card">
        <b>3. Train/Test Split</b><br>
        The dataset is split into 80% training data (used to teach the model) and 20% test data 
        (used to evaluate how well the model generalises to unseen data).
    </div>
    <div class="card">
        <b>4. Linear Regression</b><br>
        The algorithm finds the best-fit line through the data by minimising the 
        <i>Sum of Squared Errors (SSE)</i>. The model learns coefficients (weights) for each feature:<br><br>
        <code>Price = (c₁ × Square_Feet) + (c₂ × Bedrooms) + (c₃ × Bathrooms) + intercept</code>
    </div>
    <div class="card">
        <b>5. Model Evaluation</b><br>
        Three metrics are used:<br>
        &nbsp;&nbsp;• <b>R² Score</b> — how much variance the model explains (1.0 = perfect)<br>
        &nbsp;&nbsp;• <b>MAE</b> — average absolute error between predicted and actual prices<br>
        &nbsp;&nbsp;• <b>RMSE</b> — square root of average squared errors (penalises large errors more)
    </div>
    <div class="card">
        <b>6. Prediction</b><br>
        User inputs are fed into the trained model, which applies the learned equation 
        to output an estimated house price in real time.
    </div>
    <div class="card">
        <b>7. Model Persistence</b><br>
        The trained model is saved to disk using Python's <code>pickle</code> library 
        (<code>house_price_model.pkl</code>), allowing it to be reloaded without retraining.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">🚀 How to Run This Project</p>', unsafe_allow_html=True)
    st.code("""
# 1. Clone or download the project
cd house_price_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
    """, language="bash")

    st.markdown("---")
    st.markdown('<p class="section-title">🛠️ Technologies Used</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    techs = [
        ("🐍 Python", "Core programming language"),
        ("🎈 Streamlit", "Web UI framework"),
        ("🤖 scikit-learn", "ML model (LinearRegression)"),
        ("🐼 Pandas", "Data manipulation"),
        ("🔢 NumPy", "Numerical computations"),
        ("📊 Matplotlib/Seaborn", "Data visualisation"),
        ("🥒 Pickle", "Model save & load"),
        ("📁 CSV", "Dataset storage"),
    ]
    for i, (tech, desc) in enumerate(techs):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card" style="margin-bottom:0.8rem">
                <b>{tech}</b><br>
                <span style="font-size:0.85rem;color:#6b7f94">{desc}</span>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────

st.markdown("""
<div class="footer">
    Property Value Estimator &nbsp;·&nbsp; Built with Python & Streamlit &nbsp;·&nbsp; 
    scikit-learn Linear Regression &nbsp;·&nbsp; Internship-Level Project
</div>
""", unsafe_allow_html=True)