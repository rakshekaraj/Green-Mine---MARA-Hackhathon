import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. Market Prices ---
def plot_price_trends(df):
    return px.line(
        df,
        x="timestamp",
        y=["energy_price", "token_price", "hash_price"],
        labels={"value": "Price (USD)", "timestamp": "Time"},
        title="Market Prices Over Time"
    )

# --- 2. Resource Allocations ---
def plot_allocations(df):
    y_cols = ["gpu_allocated", "cpu_allocated", "fpga_allocated", "hdd_allocated", "tpu_allocated"]
    y_cols = [col for col in y_cols if col in df.columns]
    df = df.dropna(subset=["timestamp"] + y_cols)
    return px.area(df, x="timestamp", y=y_cols, title="Resource Allocations Over Time")

# --- 3. Revenue, Cost, Profit ---
def plot_revenue_profit(df):
    # y_cols = [ "cost", "profit"]
    y_cols = ["revenue", "cost", "profit"]
    df = df.dropna(subset=["timestamp"] + y_cols)
    return px.line(df, x="timestamp", y=y_cols, title="Revenue vs Cost vs Profit Over Time")

# --- 4. ESG Metrics ---
def plot_esg_metrics(df):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌍 CO₂ Emitted (kg)", round(df["co2_emitted_kg"].sum(), 2))
        st.plotly_chart(px.line(df, x="timestamp", y="co2_emitted_kg", title="CO₂ Emission Over Time"), use_container_width=True)

    with col2:
        st.metric("💧 Water Used (L)", round(df["water_used"].sum(), 2))
        st.plotly_chart(px.line(df, x="timestamp", y="water_used", title="Water Usage Over Time"), use_container_width=True)

    with col3:
        st.metric("♻️ Avg E-Waste Score", round(df["e_waste_score"].mean(), 2))
        st.plotly_chart(px.line(df, x="timestamp", y="e_waste_score", title="E-Waste Score Over Time"), use_container_width=True)

    if df["co2_emitted_kg"].sum() > 0:
        green_score = df["profit"].sum() / df["co2_emitted_kg"].sum()
        st.success(f"🌱 Green Score: {green_score:.2f} USD profit per kg CO₂")

# --- 5. Strategy Comparison ---
def plot_strategy_comparison(df):
    if "strategy" not in df.columns or "co2_emitted_kg" not in df.columns:
        return

    grouped = df.groupby(["timestamp", "strategy"]).agg({
        "profit": "sum",
        "co2_emitted_kg": "sum"
    }).reset_index()

    st.plotly_chart(px.line(grouped, x="timestamp", y="profit", color="strategy", title="💡 Strategy Comparison: Profit Over Time"), use_container_width=True)
    st.plotly_chart(px.line(grouped, x="timestamp", y="co2_emitted_kg", color="strategy", title="🌍 Strategy Comparison: CO₂ Emissions Over Time"), use_container_width=True)

# --- 6. EcoReward Optimization ---
def plot_eco_reward(df, alpha, beta, gamma, delta):
    df = df.copy()
    for col in ["profit", "co2_emitted_kg", "water_used", "e_waste_score"]:
        if col not in df:
            df[col] = 0

    df["eco_reward"] = (
        alpha * df["profit"]
        - beta * df["co2_emitted_kg"]
        - gamma * df["water_used"]
        - delta * df["e_waste_score"]
    )

    # Raw values
    df_raw = df[["timestamp", "profit", "eco_reward"]].melt(id_vars="timestamp", var_name="Metric", value_name="Value")
    fig1 = px.line(df_raw, x="timestamp", y="Value", color="Metric", title="💰 Profit vs EcoReward (Raw Values)")

    # Normalized
    df_norm = df[["timestamp", "profit", "eco_reward"]].copy()
    for col in ["profit", "eco_reward"]:
        max_val, min_val = df_norm[col].max(), df_norm[col].min()
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val) if max_val != min_val else 0
    df_norm = df_norm.melt(id_vars="timestamp", var_name="Metric", value_name="Value")
    fig2 = px.line(df_norm, x="timestamp", y="Value", color="Metric", title="📊 Profit vs EcoReward (Normalized)")

    return fig1, fig2
