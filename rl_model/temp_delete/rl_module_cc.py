# rl_module.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from state_builder import StateBuilder
import matplotlib.pyplot as plt  # Required for loss plotting
import pandas as pd

class SimpleMLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) for regression (predicting score).
    """
    def __init__(self, input_dim, hidden_dims=[64, 64]):
        super(SimpleMLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Sigmoid())  # Ensure output is in [0, 1] range
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RLScoringAgent:
    """
    Reinforcement learning scoring agent.
    This module predicts the score of inserting a candidate AV,
    and learns to regress the expected reward based on observed outcomes.
    """
    def __init__(self, traci, model_path=None, lr=5e-4, gamma=0.99):
        """
        Initialize the scoring model and training components.

        Args:
            traci: SUMO traci connection.
            model_path: Optional path to load a pretrained model.
            lr: Learning rate for optimizer. 1e-3 => 0.001; 5e-4 => 0.0005; 1e-4
            gamma: Discount factor (currently unused, but kept for future use).
        """
        self.traci = traci
        self.state_builder = StateBuilder(traci)
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.model = SimpleMLP(input_dim=8).to(self.device)
        self.model = SimpleMLP(input_dim=10).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) # for auto optimising parameters
        # self.loss_fn = nn.MSELoss() # Mean Squared Error Loss
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = []  # Buffer to store (state, reward) tuples
        self.loss_history = []  # Track training loss over time

        if model_path:
            self.load_model(model_path)

    def predict_score(self, state: np.ndarray) -> float:
        """
        Predict a scalar score for a given AV insertion state.

        Args:
            state: 8-dimensional normalized state vector.

        Returns:
            float: predicted score (higher means more suitable for insertion).
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            score = self.model(state_tensor).item()
        return score

    def record_transition(self, state: np.ndarray, reward: float):
        """
        Record one transition (state and observed reward).

        Args:
            state: state vector used in the scoring decision.
            reward: actual reward observed after AV insertion outcome.
        """
        self.memory.append((state, reward))

    def train_on_recorded(self, epochs=1):
        """
        Train the model using all recorded transitions (state, reward pairs).

        Args:
            epochs: number of training epochs per batch.
        """
        if not self.memory:
            return

        states, rewards = zip(*self.memory)
        X = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        y = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        for _ in range(epochs):
            preds = self.model(X)
            loss = self.loss_fn(preds, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_history.append(loss.item()) # Record current loss

        print(f"[Train] Fitted on {len(self.memory)} samples, final loss = {loss.item():.4f}")
        self.memory.clear()

    def plot_loss_curve(self):
        """
        Plot the loss curve based on recorded training history.
        """
        if not self.loss_history:
            print("[Plot] No loss history to show.")
            return
        plt.plot(self.loss_history, label="Training Loss", color='blue', linewidth=2, alpha=0.5)

        # Smoothed loss line (moving average)
        smoothed = pd.Series(self.loss_history).rolling(window=5).mean()
        plt.plot(smoothed, label="Smoothed Loss (window=5)", color='red', linewidth=2)

        plt.xlabel("Training Step")
        plt.ylabel("MSE Loss")
        plt.title("Loss Curve of Scoring Model")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        """
        Save the model parameters to a file.

        Args:
            path: path to save .pt model.
        """
        torch.save(self.model.state_dict(), path)
        print(f"[Model] Saved model to {path}")

    def load_model(self, path):
        """
        Load model parameters from a file.

        Args:
            path: path to load .pt model from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()  # Set to evaluation mode (disables dropout, etc.)
        print(f"[Model] Loaded model from {path}")