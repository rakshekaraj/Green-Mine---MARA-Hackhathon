import gym
import numpy as np
from gym import spaces

class MaraEnv(gym.Env):
    """
    Custom Environment for MARA Hackathon PPO agent.
    Action: [gpu_compute, asic_compute, air_miners, hydro_miners, immersion_miners, asic_miners]
    State: [energy_price, token_price, hash_price]
    """
    def __init__(self, max_power=1000000):
        super(MaraEnv, self).__init__()

        self.max_power = max_power

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([11, 11, 11, 11, 11, 11])

        self.inventory = {
            "gpu_compute": {"power": 5000, "esg": (0.7, 1, 2)},
            "asic_compute": {"power": 15000, "esg": (0.9, 0.5, 3)},
            "air_miners": {"power": 3500, "esg": (1.2, 2, 1)},
            "hydro_miners": {"power": 5000, "esg": (0.3, 10, 2)},
            "immersion_miners": {"power": 10000, "esg": (1.8, 5, 3)},
            "asic_miners": {"power": 15000, "esg": (0.9, 0.5, 3)}
        }

        self.alpha = 1.0
        self.beta = 1000.0
        self.gamma = 500.0
        self.delta = 100.0

        self.reset()

    def step(self, action):
        
        names = list(self.inventory.keys())
        alloc = {names[i]: int(action[i]) for i in range(6)}
        self.current_allocation = alloc  # ✅ store for external API use

        # Power constraint
        total_power = sum(alloc[k] * self.inventory[k]['power'] for k in alloc)
        if total_power > self.max_power:
            self.last_carbon = self.last_water = self.last_ewaste = 0
            return self.state, -1e6, True, {"reason": "Power limit exceeded"}

        energy_price, token_price, hash_price = self.state

        # 💸 Revenue with noise to break mirroring
        revenue = sum(
            alloc[k] * token_price * np.random.uniform(8, 12) if "compute" in k 
            else alloc[k] * hash_price * np.random.uniform(4, 6)
            for k in alloc
        )

        cost = total_power + energy_price
        profit = revenue - cost
        revenue = profit + cost

        # 🌱 ESG metrics
        co2 = sum(alloc[k] * self.inventory[k]['esg'][0] for k in alloc)
        water = sum(alloc[k] * self.inventory[k]['esg'][1] for k in alloc)
        ewaste = sum(alloc[k] * self.inventory[k]['esg'][2] for k in alloc)

        self.last_carbon = co2
        self.last_water = water
        self.last_ewaste = ewaste

        reward = self.alpha * profit - self.beta * co2 - self.gamma * water - self.delta * ewaste

        self.state = self._sample_prices()
        return self.state, reward, False, {
            "revenue": revenue,
            "profit": profit,
            "cost": cost,
            "power_used": total_power,
            "energy_price": self.state[0],
            "token_price": self.state[1],
            "hash_price": self.state[2],
            "eco_reward": reward,
            "raw_profit": profit
        }

        # return self.state, reward, False, {"raw_profit": profit}


    def reset(self):
        self.state = self._sample_prices()
        self.current_allocation = {
            "gpu_compute": 0, "asic_compute": 0, "air_miners": 0,
            "hydro_miners": 0, "immersion_miners": 0, "asic_miners": 0
        }
        self.last_carbon = self.last_water = self.last_ewaste = 0
        return self.state

    def _sample_prices(self):
        # Expanded price variability for realism
        energy_price = np.random.uniform(0.4, 2.5)
        token_price = np.random.uniform(1.5, 6.0)
        hash_price = np.random.uniform(3.0, 15.0)
        return np.array([energy_price, token_price, hash_price], dtype=np.float32)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
