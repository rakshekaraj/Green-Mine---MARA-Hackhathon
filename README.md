# GreenMine: Real-Time ESG-Aware Trading Dashboard

GreenMine is a real-time reinforcement learning (RL) dashboard developed for the MARA Hackathon 2025. It optimizes the allocation of compute resources (e.g., ASICs, GPUs, immersion miners) to maximize financial profit while minimizing environmental impact — specifically CO₂ emissions, water consumption, and electronic waste (e-waste).

This system integrates a custom RL environment, live market data, real-time ESG signals, and a Streamlit-based dashboard for tuning and visualization.

---

## 🌍 Key Features

- ✅ Real-time resource allocation via PPO agent.
- 📉 Live visualization of market trends, emissions, and optimization scores.
- 🔁 Dynamic adjustment of reward weights (profit, CO₂, water, e-waste).
- ⚙️ Modular Gym environment with realistic constraints.
- 🧠 Strategy comparison between reinforcement learning and baseline heuristics.
- 🔄 30-second auto-refresh with current price and ESG signals.

---

## 📺 Live Demo

> Coming soon…

---

## ⚙️ Tech Stack

- **Python** (Gym, NumPy, Pandas)
- **Stable Baselines3 (PPO)**
- **Streamlit** (Dashboard UI)
- **Plotly** (Interactive charts)
- **APIs**:
  - [MARA Hackathon API](https://mara-hackathon-api.onrender.com) for real-time prices and submission
  - [Electricity Maps API](https://www.electricitymap.org/) for carbon intensity

---

## Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/greenmine-esg-dashboard.git
cd greenmine-esg-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the agent loop (in background)
python agent.py

# 4. Launch the dashboard
streamlit run streamlit_app.py
