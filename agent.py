import os
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from stable_baselines3 import PPO
from mara_env import MaraEnv  # Make sure mara_env.py is in the same folder

# ==== CONFIG ====
API_KEY = "e722ea9c-e440-486a-ac6f-344145bbfff4"
SITE_NAME = "EcoTradeAI"
MODEL_PATH = "ppo_mara_agent.zip"
METRICS_CSV = "data/metrics.csv"
API_URL = "https://mara-hackathon-api.onrender.com/machines"

# ==== ENV + MODEL ====
env = MaraEnv()
model = PPO.load(MODEL_PATH)

# ==== METRICS FILE INIT ====
if not os.path.exists(METRICS_CSV):
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(columns=["timestamp", "profit", "carbon_emitted", "water_used", "e_waste_score"]).to_csv(METRICS_CSV, index=False)

# ==== SEND ALLOCATION ====
def send_allocation(allocation_list):
    headers = {"X-Api-Key": API_KEY, "Content-Type": "application/json"}
    allocation_dict = {
        "asic_compute": int(allocation_list[0]),
        "gpu_compute": int(allocation_list[1]),
        "air_miners": int(allocation_list[2]),
        "hydro_miners": int(allocation_list[3]),
        "immersion_miners": int(allocation_list[4])
    }
    try:
        res = requests.put(API_URL, json=allocation_dict, headers=headers)
        print("🔁 Response:", res.status_code, res.text)
        return res.json() if res.ok else None
    except Exception as e:
        print(f"❌ Error during allocation: {e}")
        return None

# ==== AGENT LOOP ====
while True:
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # Convert action to native Python ints and send allocation
        result = send_allocation(action)

        if result:
            now = datetime.utcnow().isoformat()
            # ✅ log raw financial profit, not the ESG-penalized reward
            raw_profit = info.get("raw_profit", reward)
            new_row = {
                "timestamp": now,
                "energy_price": info["energy_price"],
                "token_price": info["token_price"],
                "hash_price": info["hash_price"],
                "gpu_compute": int(action[1]),
                "asic_compute": int(action[0]),
                "air_miners": int(action[2]),
                "hydro_miners": int(action[3]),
                "immersion_miners": int(action[4]),
                "asic_miners": int(action[5]),
                "power_used": info["power_used"],
                "power_cost": info["cost"],
                "revenue": info["revenue"],
                "profit": info["profit"],
                "carbon_emitted": env.last_carbon,
                "water_used": env.last_water,
                "e_waste_score": env.last_ewaste,
                "eco_reward": info["eco_reward"],
                "strategy": "LiveRL"
            }

            pd.DataFrame([new_row]).to_csv(METRICS_CSV, mode='a', header=False, index=False)
            print(f"✅ Logged step: {new_row}")
        else:
            print("⚠️ Allocation failed.")

        time.sleep(10)
