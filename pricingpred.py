import requests
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ========== 1. Fetch and prepare data ==========
def fetch_price_data():
    url = "https://mara-hackathon-api.onrender.com/prices"
    data = requests.get(url).json()
    # Reverse to ensure chronological order (oldest to newest)
    data = list(reversed(data))
    return np.array([
        [d['energy_price'], d['hash_price'], d['token_price']]
        for d in data
    ], dtype=np.float32)

# ========== 2. Dataset prep ==========
def create_dataset(data, window_size=10, horizon=2):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ========== 3. Model ==========
class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2 * input_size)  # 2 timesteps × 3 features

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])  # shape: (batch, 6)
        return out.view(-1, 2, 3)  # shape: (batch, 2, 3)

# ========== 4. Training ==========
def train_model(X, y, epochs=100):
    model = LSTMPricePredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    return model

# ========== 5. Inference ==========
def predict_next_two_steps(model, recent_scaled_window, scaler):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(recent_scaled_window[-10:], dtype=torch.float32).unsqueeze(0)  # shape: (1, 10, 3)
        pred_scaled = model(input_seq).squeeze(0).numpy()  # shape: (2, 3)
        return scaler.inverse_transform(pred_scaled)

# ========== 6. Run everything ==========
def main_train_and_predict():
    # Load and preprocess data
    raw_data = fetch_price_data()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(raw_data)

    # Prepare dataset
    X, y = create_dataset(scaled_data, window_size=10, horizon=2)

    # Train
    model = train_model(X, y, epochs=100)

    # Save model and scaler
    torch.save(model.state_dict(), "price_predictor.pth")
    np.save("scaler_min.npy", scaler.data_min_)
    np.save("scaler_max.npy", scaler.data_max_)

    # Predict next 2 steps
    pred = predict_next_two_steps(model, scaled_data, scaler)
    print("\nPredicted next 2 steps:")
    print("Step 1 -> Energy: {:.4f}, Hash: {:.4f}, Token: {:.4f}".format(*pred[0]))
    print("Step 2 -> Energy: {:.4f}, Hash: {:.4f}, Token: {:.4f}".format(*pred[1]))

# ========== 7. Online Inference Function ==========
def load_model_and_predict(fine_tune=False, fine_tune_epochs=3):
    # Load model
    model = LSTMPricePredictor()
    model.load_state_dict(torch.load("price_predictor.pth"))

    # Load scaler
    scaler = MinMaxScaler()
    scaler.min_ = np.zeros(3)
    scaler.scale_ = 1 / (np.load("scaler_max.npy") - np.load("scaler_min.npy"))
    scaler.data_min_ = np.load("scaler_min.npy")
    scaler.data_max_ = np.load("scaler_max.npy")

    # Fetch latest data
    raw_data = fetch_price_data()
    recent_data = np.array([
        [d['energy_price'], d['hash_price'], d['token_price']]
        for d in reversed(raw_data)
    ], dtype=np.float32)

    scaled_data = scaler.transform(recent_data)

    # Optional fine-tuning
    if fine_tune and len(scaled_data) >= 12:
        X_ft, y_ft = create_dataset(scaled_data[-20:], window_size=10, horizon=2)
        X_ft_tensor = torch.tensor(X_ft, dtype=torch.float32)
        y_ft_tensor = torch.tensor(y_ft, dtype=torch.float32)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.MSELoss()

        for _ in range(fine_tune_epochs):
            optimizer.zero_grad()
            pred = model(X_ft_tensor)
            loss = criterion(pred, y_ft_tensor)
            loss.backward()
            optimizer.step()

        # Save updated weights
        torch.save(model.state_dict(), "price_predictor.pth")

    # Inference using last 10 points
    recent_window = scaled_data[-10:]
    prediction = predict_next_two_steps(model, recent_window, scaler)

    return prediction


if __name__ == '__main__':
    main_train_and_predict()

# Uncomment to run live inference only (after training)
# predictions = load_model_and_predict()
# print("Live Prediction:")
# for i, p in enumerate(predictions):
#     print(f"Step {i+1} -> Energy: {p[0]:.4f}, Hash: {p[1]:.4f}, Token: {p[2]:.4f}")

## Online fine-tuning
# predictions = load_model_and_predict(fine_tune=True)
## No fine-tuning
# predictions = load_model_and_predict(fine_tune=False)
