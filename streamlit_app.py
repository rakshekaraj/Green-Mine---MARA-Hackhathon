import streamlit as st
import pandas as pd
import requests
from utils import charts
from streamlit_autorefresh import st_autorefresh
import numpy as np

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="GreenMine Dashboard", layout="wide")
st.title("🔋 The GreenMine — Real-Time ESG-Aware Trading Dashboard")

# Sidebar sliders
st.sidebar.markdown("### EcoReward Weights (Normalized Scale)")
alpha = st.sidebar.slider("α (Normalized Profit Weight)", 0.0, 2.0, 1.0)
beta = st.sidebar.slider("β (Normalized CO2 Penalty)", 0.0, 2.0, 1.0)
gamma = st.sidebar.slider("γ (Water Penalty)", 0.0, 2.0, 0.5)
delta = st.sidebar.slider("δ (E-Waste Penalty)", 0.0, 2.0, 0.5)

# ---------------------- API HELPERS ----------------------
@st.cache_data(ttl=300)
def get_carbon_intensity(zone="US-CAL-CISO"):
    url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"
    headers = {"auth-token": "mJQ7MRxitUF6BykPBXcm"}  # Replace with actual token
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["carbonIntensity"] if isinstance(data["carbonIntensity"], (int, float)) else data["carbonIntensity"]["carbonIntensity"]
    except Exception as e:
        st.error(f"Failed to fetch carbon intensity: {e}")
        return 0

@st.cache_data(ttl=60)
def fetch_market_prices():
    res = requests.get("https://mara-hackathon-api.onrender.com/prices")
    print(res.status_code, res.json())

    if res.status_code == 200:
        return pd.DataFrame(res.json())
    else:
        st.warning("⚠️ Failed to fetch market prices.")
        return pd.DataFrame()

# ---------------------- DATA COMBINATION ----------------------
@st.cache_data(ttl=60)
def load_combined_live_data():
    df = fetch_market_prices()
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["strategy"] = "LiveRL"
    df["compute_allocated"] = np.random.randint(180, 320, size=len(df))
    df["power"] = df["compute_allocated"] *np.random.uniform(0.5, 1.5, size=len(df)) # watts


    if df["token_price"].std() < 1e-3:  # or some low threshold
        df["token_price"] = np.random.uniform(2.5, 5.5, size=len(df))

    if df["token_price"].std() < 0.01:
        df["unit_price"] = np.random.uniform(1.5, 7.5, size=len(df))
    else:
        df["unit_price"] = df["token_price"] * np.random.uniform(0.9, 1.1, size=len(df))
    st.line_chart(df[["timestamp", "token_price"]].set_index("timestamp"))

    df["cost"] = df["energy_price"] * df["power"]
    df["revenue"] = df["unit_price"] * df["compute_allocated"] 


    df["profit"] = df["revenue"] - df["cost"]
# revenue to profit changed

    # Carbon intensity and emissions
    ci = get_carbon_intensity()
    st.sidebar.markdown(f"**Live Carbon Intensity**: {ci} gCO₂/kWh")
    df["power_kwh"] = df["power"] / 1000
    df["carbon_intensity"] = ci + np.random.uniform(-50, 50, size=len(df))
    df["co2_emitted_kg"] = (df["power_kwh"] * df["carbon_intensity"]) / 1000

    # ESG Metrics
    df["water_used"] = df["power_kwh"] * 0.1 * np.random.uniform(0.9, 1.1, size=len(df))
    df["e_waste_score"] = df["compute_allocated"] * np.random.uniform(0.001, 0.01, size=len(df))

    # Resource allocation samples
    df["gpu_allocated"] = np.random.randint(10, 100, size=len(df))
    df["cpu_allocated"] = np.random.randint(5, 50, size=len(df))
    df["fpga_allocated"] = np.random.randint(0, 20, size=len(df))
    df["hdd_allocated"] = np.random.randint(20, 100, size=len(df))
    df["tpu_allocated"] = np.random.randint(0, 10, size=len(df))

    return df

# ---------------------- DISPLAY ----------------------
st.subheader("📡 Live MARA Market Prices")
prices_df = fetch_market_prices()
st.dataframe(prices_df.tail(10), use_container_width=True)

try:
    data = load_combined_live_data()
except Exception as e:
    st.error(f"❌ Data failed to load: {e}")
    data = pd.DataFrame()

if data.empty:
    st.stop()

strategy = st.sidebar.selectbox("Strategy", ["All"] + sorted(data["strategy"].dropna().unique()))
if strategy != "All":
    data = data[data["strategy"] == strategy]

# ---------------------- PLOTS ----------------------
st.header("📈 Price Trends")
st.plotly_chart(charts.plot_price_trends(data), use_container_width=True)

st.header("⚙️ Resource Allocations")
st.plotly_chart(charts.plot_allocations(data), use_container_width=True)

st.header("💸 Revenue & Profit")
st.plotly_chart(charts.plot_revenue_profit(data), use_container_width=True)

st.header("🌱 ESG Metrics")
charts.plot_esg_metrics(data)

st.header("🌿 EcoReward Optimization Score")
fig1, fig2 = charts.plot_eco_reward(data, alpha, beta, gamma, delta)
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

st.header("🧠 Strategy Comparison")
charts.plot_strategy_comparison(data)

if "co2_emitted_kg" in data.columns:
    st.subheader("💸 Profit vs 🌿 CO₂ Emissions")
    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(data[["timestamp", "profit"]].set_index("timestamp"))
    with col2:
        st.line_chart(data[["timestamp", "co2_emitted_kg"]].set_index("timestamp"))

st_autorefresh(interval=30 * 1000, key="refresh")
