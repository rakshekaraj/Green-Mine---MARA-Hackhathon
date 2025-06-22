from stable_baselines3 import PPO
from mara_env import MaraEnv

# Step 1: Initialize environment
env = MaraEnv()

# Step 2: Define PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=256,
    batch_size=64,
    learning_rate=1e-4,
    device="auto"  # runs on M2 CPU
)

# Step 3: Train
model.learn(total_timesteps=10000)  # start small (you can increase later)

# Step 4: Save
model.save("ppo_mara_agent")
print("✅ PPO agent trained and saved.")
