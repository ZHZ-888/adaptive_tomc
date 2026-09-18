# rl_module.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import os

# Slurm compute nodes normally have no display server. Use a non-interactive
# backend there so importing pyplot can never require X11/Tk.
if os.environ.get("RUN_DIR"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # Required for local plotting
import pandas as pd
import csv
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

from rl_model.state_builder import StateBuilder

class SimpleMLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) for regression (predicting score).
    """
    def __init__(self, input_dim, hidden_dims=(64, 64)):
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
    def __init__(self, traci, data_recorder, exp_name, model_path=None,
                 lr=5e-4, gamma=0.99, hidden_dims=(64, 64)):
        """
        Initialize the scoring model and training components.

        Args:
            traci: SUMO traci connection.
            model_path: Optional path to load a pretrained model.
            lr: Learning rate for optimizer. 1e-3 => 0.001; 5e-4 => 0.0005; 1e-4
            gamma: Discount factor (currently unused, but kept for future use).
        """
        self.traci = traci
        self.data_recorder = data_recorder
        self.state_builder = StateBuilder(traci, data_recorder)
        self.gamma = gamma
        self.device = torch.device("cpu")  # Force CPU for compatibility and simplicity

        self.model = SimpleMLP(input_dim=7, hidden_dims=list(hidden_dims)).to(self.device)
        print(f"***** Learning rate: {lr} *****")
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr) # for auto optimising parameters
        # self.loss_fn = nn.MSELoss() # Mean Squared Error Loss
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = []  # Buffer to store (state, reward) tuples
        self.loss_history = []  # (global_epoch, simulation_step, epoch_loss)
        self.training_epoch = 0
        self.training_session = 0
        self.score_log_index = 0

        self.IS_HPC = "RUN_DIR" in os.environ  # Simple check for HPC environment variable
        # **** Define project root (rl_model folder) ****
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # Root directory for logs and models (HPC-aware)
        run_root = os.environ.get("RUN_DIR", os.path.join(self.base_dir, "rl_logs")) # environ (environment)

        # Create a unique folder name for this exp (e.g. CA_LR0.0005_B16_E5_S21_20260422_1830)
        array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        suffix = f"_task{array_id}" if array_id else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.run_id = f"{exp_name}_{timestamp}_{suffix}"
        # All-in-one run directory for logs AND models
        self.run_dir = os.path.join(run_root, self.run_id)

        os.makedirs(self.run_dir, exist_ok=True)
        # Model saving directory (optional: same as logs or a subfolder)
        self.model_save_dir = os.path.join(self.run_dir, "models")
        os.makedirs(self.model_save_dir, exist_ok=True)
        # Initialise TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=self.run_dir)
        print(f"[TensorBoard] Logging to {self.run_dir}")

        if model_path: # predict mode, load the model
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

    def train_on_recorded(self, current_step, epochs=5, batch_size=16):
        """
        Train the model using all recorded transitions (state, reward pairs).
        Args:
            epochs: number of training epochs per batch.
            batch_size: size of each training mini-batch.
        """
        if not self.memory:
            return

        states, rewards = zip(*self.memory)
        X_all = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        y_all = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset_size = len(X_all)
        indices = np.arange(dataset_size)

        total_loss = 0.0
        batch_count = 0

        for _ in range(epochs):
            epoch_loss = 0.0
            epoch_batch_count = 0
            np.random.shuffle(indices) # Shuffle the data at each epoch
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                X_batch = X_all[batch_idx]
                y_batch = y_all[batch_idx]

                preds = self.model(X_batch)
                loss = self.loss_fn(preds, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_value = loss.item()
                epoch_loss += loss_value
                epoch_batch_count += 1
                total_loss += loss_value
                batch_count += 1

            avg_epoch_loss = epoch_loss / epoch_batch_count
            self.training_epoch += 1
            self.loss_history.append(
                (self.training_epoch, current_step, avg_epoch_loss)
            )
            self._append_csv_row(
                "loss_log.csv",
                ("epoch", "sim_step", "loss"),
                (self.training_epoch, current_step, avg_epoch_loss)
            )
            self.writer.add_scalar(
                'Loss/Epoch_Training_Loss', avg_epoch_loss,
                self.training_epoch
            )

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        self.writer.add_scalar('Loss/Avg_Training_Loss', avg_loss, current_step)
        print(f"[Train] Fitted on {len(self.memory)} samples, at step {current_step}, Avg_loss = {avg_loss:.4f}")
        self.memory.clear()
        self.training_session += 1

    def _append_csv_row(self, filename, fieldnames, values):
        """Append one durable log row, writing the header only once."""
        file_path = os.path.join(self.run_dir, filename)
        file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0
        with open(file_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(fieldnames)
            writer.writerow(values)

    def log_score(self, score):
        """Persist a predicted score as soon as the decision is made."""
        self._append_csv_row(
            "score_log.csv", ("index", "score"),
            (self.score_log_index, float(score))
        )
        self.score_log_index += 1

    def log_score_reward(self, vehicle_id, score, reward):
        """Persist a completed delayed score/reward pair immediately."""
        self._append_csv_row(
            "score_reward_log.csv",
            ("vehicle_id", "score", "reward"),
            (vehicle_id, float(score), float(reward))
        )

    def log_training_metrics(self, current_step):
        """
        SAVE (Records) training performance data to a CSV file.
        Triggered every time a training session is initiated.
        """
        if not self.memory:
            return

        recent_samples = self.memory
        recent_rewards = [m[1] for m in recent_samples]

        avg_reward = np.mean(recent_rewards)
        max_reward = np.max(recent_rewards)
        min_reward = np.min(recent_rewards)

        self.writer.add_scalar('Reward/Average', avg_reward, current_step)
        self.writer.add_scalar('Reward/Max', max_reward, current_step)
        self.writer.add_scalar('Reward/Min', min_reward, current_step)

        log_entry = {
            'session_id': self.training_session,
            'sim_step': current_step,  # Current simulation time-step
            'avg_reward': round(float(avg_reward), 4),
            'max_reward': round(float(max_reward), 4),
            'min_reward': round(float(min_reward), 4),
            'sample_size': len(recent_samples)}

        file_path = os.path.join(self.run_dir, "reward_log.csv")
        file_exists = os.path.isfile(file_path)

        try:
            df = pd.DataFrame([log_entry])
            df.to_csv(file_path, mode='a', header=not file_exists, index=False)
            print(f"[Log] Session {log_entry['session_id']} at Step {current_step}: "
                  f"Mean Reward = {avg_reward:.3f}")

        except Exception as e:
            print(f"[Error] Failed to log training metrics: {e}")

    def record_plot_loss(self):
        """
        SAVE loss history CSV and plot the loss curve based on recorded training history.
        """
        if not self.loss_history:
            print("[Plot] No loss history to show.")
            return
        epochs, simulation_steps, losses = zip(*self.loss_history)
        loss_csv_path = os.path.join(self.run_dir, "loss_log.csv")
        df_loss = pd.DataFrame({
            'epoch': epochs,
            'sim_step': simulation_steps,
            'loss': losses
        })
        df_loss.to_csv(loss_csv_path, index=False)
        print(f"[Plot] Loss data saved to {loss_csv_path}")

        if not self.IS_HPC:
            plt.plot(epochs, losses, label="Epoch Loss", color='blue', linewidth=1, alpha=0.3)
            loss_series = pd.Series(losses)
            smoothed = loss_series.rolling(window=10).mean()
            plt.plot(epochs, smoothed, label="Smoothed Loss (window=10)", color='red', linewidth=2)
            plt.xlabel("Training Epoch")
            plt.ylabel("Smooth L1 Loss")
            plt.title("Loss Curve of Scoring Model")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def record_plot_scores(self, ls_score):
        """
        Save score history CSV and plot score distribution.
        """
        score_csv_path = os.path.join(self.run_dir, "score_log.csv")
        df_scores = pd.DataFrame({'index': list(range(len(ls_score))), 'score': ls_score})
        df_scores.to_csv(score_csv_path, index=False)
        if not self.IS_HPC:
            plt.figure(figsize=(8, 4))
            plt.scatter(range(len(ls_score)), ls_score, s=3, c='blue', label='Score', alpha=0.5)
            plt.title('Score Distribution per Decision')
            plt.xlabel('Decision Index')
            plt.ylabel('Predicted Score')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
    def record_plot_score_reward(self, dic_score_reward):
        """Save and plot paired decision scores and realised rewards."""
        paired = [
            (vehicle_id, values[0], values[1])
            for vehicle_id, values in dic_score_reward.items()
            if len(values) >= 2
        ]
        if not paired:
            print("[Plot] No completed score-reward pairs.")
            return

        vehicle_ids, scores, rewards = zip(*paired)
        df = pd.DataFrame({
            "vehicle_id": vehicle_ids,
            "score": scores,
            "reward": rewards,
        })
        csv_path = os.path.join(self.run_dir, "score_reward_log.csv")
        df.to_csv(csv_path, index=False)
        print(f"[Plot] Score-reward data saved to {csv_path}")

        score_array = np.asarray(scores, dtype=float)
        reward_array = np.asarray(rewards, dtype=float)
        if (len(score_array) >= 2 and
                np.std(score_array) > 0 and np.std(reward_array) > 0):
            correlation = float(np.corrcoef(score_array, reward_array)[0, 1])
            correlation_text = f"{correlation:.3f}"
        else:
            correlation_text = "N/A"
        print(f"[Metric] Score-reward correlation: {correlation_text}")

        if not self.IS_HPC:
            colours = ["red" if reward <= 0 else "blue" for reward in rewards]
            plt.figure(figsize=(6, 5))
            plt.scatter(scores, rewards, c=colours, s=15, alpha=0.6)
            plt.xlabel("Predicted Score")
            plt.ylabel("Realised Reward")
            plt.title(
                "Predicted Score vs Realised Reward\n"
                f"Pearson correlation = {correlation_text}"
            )
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    def save_model(self, filename):
        """
        Save the model parameters to a file.
        """
        full_path = os.path.join(self.model_save_dir, filename)
        torch.save(self.model.state_dict(), full_path)
        print(f"[Model] Saved model to {full_path}")

    def load_model(self, path):
        """
        Load model parameters from a file.

        Args:
            path: path to load .pt model from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()  # Set to evaluation mode (disables dropout, etc.)
        print(f"[Model] Loaded model from {path}")


class GateMLP(nn.Module):
    """
    Shared dispatch network.

    Input:
        [
            q_top,
            delta_q,
            n_candidate_norm,
            d_target_to_MCZ_norm,
            phi_target,
            expert_type
        ]

    Output:
        [reject_logit, execute_logit]
    """

    def __init__(self, input_dim=6, hidden_dims=(16, 16)):
        super(GateMLP, self).__init__()

        layers = []
        dims = [input_dim] + list(hidden_dims) + [2]

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SelfGateAgent:
    """
    Task-conditioned self-gating agent.

    This agent does NOT rank candidate AVs.
    It only decides whether the top-ranked candidate selected by the score model
    should be executed or rejected.
    """

    def __init__(self, exp_name, model_path=None, lr=5e-4, input_dim=6,
                 hidden_dims=(16, 16)):
        self.device = torch.device("cpu")  # Keep consistent with RLScoringAgent

        self.model = GateMLP(input_dim=input_dim, hidden_dims=hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.memory = []
        self.loss_history = []
        self.acc_history = []
        self.training_update = 0

        # Same log style as RLScoringAgent
        self.IS_HPC = "RUN_DIR" in os.environ
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        run_root = os.environ.get("RUN_DIR", os.path.join(self.base_dir, "rl_logs"))

        array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "")
        suffix = f"_task{array_id}" if array_id else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.run_id = f"{exp_name}_{timestamp}_{suffix}"
        self.run_dir = os.path.join(run_root, self.run_id)

        os.makedirs(self.run_dir, exist_ok=True)

        self.model_save_dir = os.path.join(self.run_dir, "models")
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.run_dir)
        print(f"[Gate TensorBoard] Logging to {self.run_dir}")

        run_root = os.environ.get("RUN_DIR", self.run_dir)
        os.makedirs(run_root, exist_ok=True)
        self.gate_training_csv = os.path.join(run_root, "gate_training_log.csv")
        self.gate_transition_csv = os.path.join(run_root, "gate_transition_log.csv")
        if os.path.isfile(self.gate_training_csv):
            try:
                old_log = pd.read_csv(
                    self.gate_training_csv,
                    usecols=["training_update"]
                )
                if not old_log.empty:
                    self.training_update = int(
                        old_log["training_update"].max()
                    )
            except (ValueError, pd.errors.EmptyDataError):
                self.training_update = 0

        if model_path:
            self.load_model(model_path)

    def build_gate_input(self,
            scores,
            top_idx,
            task_name,
            d_target_to_MCZ,
            target_platoon_size=None,
            n_free=None,
            n_follower=None,
            max_platoon_size=12,
            length_pfz=1200,
            max_candidates=10,
    ):
        """
        Build the six-feature dispatch input:

            [q_top,
            delta_q,
            n_candidate_norm,
            d_target_to_MCZ_norm,
            phi_target,
            expert_type]

        expert_type:
            0.0 -> collecting expert (CE)
            1.0 -> splitting expert (SE)
        """
        scores = np.asarray(scores,
            dtype=np.float32).reshape(-1)

        if scores.size == 0:
            raise ValueError("[SelfGateAgent] scores cannot be empty.")

        if top_idx < 0 or top_idx >= scores.size:
            raise IndexError(f"[SelfGateAgent] Invalid top_idx={top_idx} "
                f"for {scores.size} scores.")

        # ===== Raw features =====
        # 1. Highest score
        q_top = float(scores[top_idx])

        # 2. Score margin
        if scores.size >= 2:
            sorted_scores = np.sort(scores)
            q_second = float(sorted_scores[-2])
            delta_q = q_top - q_second
        else:
            delta_q = 0.0

        # 3. Number of valid candidates
        n_candidate = int(scores.size)

        # 4. Remaining distance of target leader to MCZ
        d_target_to_MCZ = float(d_target_to_MCZ)

        # 5 and 6. Target abnormality and expert type
        if task_name == "collecting":
            if n_free is None or n_follower is None:
                raise ValueError("[SelfGateAgent] CE requires "
                    "n_free and n_follower.")
            phi_target = (float(n_free) / max(float(n_follower), 1.0))
            expert_type = 0.0

        elif task_name == "splitting":
            if target_platoon_size is None:
                raise ValueError("[SelfGateAgent] SE requires "
                    "target_platoon_size.")
            phi_target = ((float(target_platoon_size) - float(max_platoon_size))
                          / max(float(max_platoon_size), 1.0))
            expert_type = 1.0

        else:
            raise ValueError(f"[SelfGateAgent] Unknown task_name: "
                f"{task_name}")

        # ===== Normalisation =====
        q_top_norm = float(np.clip(q_top,0.0,1.0))
        delta_q_norm = float(np.clip(delta_q,0.0,1.0))
        n_candidate_norm = float(np.clip(n_candidate / max(float(max_candidates), 1.0),
            0.0,1.0))
        d_target_to_MCZ_norm = float(np.clip(
            d_target_to_MCZ / max(float(length_pfz), 1.0),
            0.0,1.0))
        phi_target_norm = float(np.clip(phi_target,
            0.0,1.0))

        # ===== Final gate input =====
        gate_input = np.array([
            q_top_norm,
            delta_q_norm,
            n_candidate_norm,
            d_target_to_MCZ_norm,
            phi_target_norm,
            expert_type
        ], dtype=np.float32)

        return gate_input

    def predict_execute(self, gate_input):
        """
        Predict execute/reject decision.

        Returns
        -------
        execute_decision : bool
            True means execute top AV.
            False means reject / no action.

        logits_np : np.ndarray
            [reject_logit, execute_logit]

        probs_np : np.ndarray
            Softmax probabilities.
        """
        self.model.eval()

        x = torch.tensor(
            gate_input,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x).squeeze(0)
            probs = torch.softmax(logits, dim=0)

        logits_np = logits.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()

        execute_decision = logits_np[1] > logits_np[0]

        return execute_decision, logits_np, probs_np

    def record_transition(self, gate_input, reward, task_name):
        """
        Record one delayed outcome sample for gate training.

        Labels and sample weights are assigned in train_on_recorded():
            reward > 0  -> execute
            reward <= 0 -> defer, weight = 1.0
            sample weight = abs(reward)
        """
        if task_name not in ("collecting", "splitting"):
            raise ValueError(f"[SelfGateAgent] Unknown task_name: {task_name}")

        gate_input = np.asarray(gate_input, dtype=np.float32).reshape(-1)
        reward = float(reward)

        self.memory.append({
            "gate_input": gate_input,
            "reward": reward,
            "task_name": task_name
        })

        transition_row = {
            "task_name": task_name,
            "q_top": float(gate_input[0]),
            "delta_q": float(gate_input[1]),
            "n_candidate_norm": float(gate_input[2]),
            "d_target_to_MCZ_norm": float(gate_input[3]),
            "phi_target": float(gate_input[4]),
            "expert_type": float(gate_input[5]),
            "reward": reward,
            "label": int(reward > 0),
            "sample_weight": 1.0 if reward <= 0 else abs(reward),
        }
        file_exists = os.path.isfile(self.gate_transition_csv)
        pd.DataFrame([transition_row]).to_csv(
            self.gate_transition_csv,
            mode="a",
            header=not file_exists,
            index=False
        )

    def train_on_recorded(self, current_step, epochs=5, batch_size=16):
        """
        Train gate model using recorded delayed outcome samples.

        TSG labels and weights are generated from raw reward:
            raw_reward > 0  -> execute
            raw_reward <= 0 -> defer/reject, weight = 1.0
            sample weight = abs(raw_reward)
        """
        if not self.memory:
            return

        X = np.array([m["gate_input"] for m in self.memory], dtype=np.float32)

        raw_rewards = np.array(
            [m["reward"] for m in self.memory],
            dtype=np.float32
        )

        # The reward sign determines the execute/defer label.
        y = (raw_rewards > 0).astype(np.int64)

        # Reward magnitude determines the importance of each sample.
        reward_weights = np.where(
            raw_rewards <= 0,
            1.0,  # Failure: strongest defer feedback
            raw_rewards  # Success: weighted by insertion quality
        ).astype(np.float32)

        X_all = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_all = torch.tensor(y, dtype=torch.long).to(self.device)
        reward_weights_all = torch.tensor(
            reward_weights,
            dtype=torch.float32
        ).to(self.device)

        dataset_size = len(X_all)
        indices = np.arange(dataset_size)

        n_reject = int(np.sum(y == 0))
        n_execute = int(np.sum(y == 1))

        loss_fn = nn.CrossEntropyLoss(reduction="none")

        total_loss = 0.0
        total_acc = 0.0
        batch_count = 0
        epoch_metrics = []

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_batch_count = 0
            epoch_weighted_correct = 0.0
            epoch_reward_weight = 0.0

            np.random.shuffle(indices)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                X_batch = X_all[batch_idx]
                y_batch = y_all[batch_idx]
                reward_weight_batch = reward_weights_all[batch_idx]

                logits = self.model(X_batch)

                sample_losses = loss_fn(logits, y_batch)

                loss = (
                               sample_losses * reward_weight_batch
                       ).sum() / reward_weight_batch.sum().clamp_min(1e-8)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred = torch.argmax(logits, dim=1)
                correct = (pred == y_batch).float()
                acc = correct.mean().item()

                epoch_weighted_correct += (
                        correct * reward_weight_batch
                ).sum().item()

                epoch_reward_weight += reward_weight_batch.sum().item()

                epoch_loss += loss.item()
                epoch_acc += acc
                epoch_batch_count += 1

                total_loss += loss.item()
                total_acc += acc
                batch_count += 1

            self.training_update += 1

            epoch_weighted_acc = (
                    epoch_weighted_correct
                    / max(epoch_reward_weight, 1e-8)
            )

            epoch_metrics.append({
                "training_update": self.training_update,
                "current_step": current_step,
                "epoch": epoch + 1,
                "loss": epoch_loss / epoch_batch_count,
                "accuracy": epoch_acc / epoch_batch_count,
                "reward_weighted_accuracy": epoch_weighted_acc,
            })

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        avg_acc = total_acc / batch_count if batch_count > 0 else 0.0
        avg_raw_reward = float(np.mean(raw_rewards)) if len(raw_rewards) > 0 else 0.0
        avg_feedback_weight = float(np.mean(reward_weights)) if len(reward_weights) > 0 else 0.0

        for row in epoch_metrics:
            update = row["training_update"]

            self.loss_history.append((update, row["loss"]))
            self.acc_history.append((update, row["accuracy"]))

            self.writer.add_scalar("Gate/Loss", row["loss"], update)
            self.writer.add_scalar("Gate/Accuracy", row["accuracy"], update)
            self.writer.add_scalar(
                "Gate/Reward_Weighted_Accuracy",
                row["reward_weighted_accuracy"],
                update
            )

        log_step = self.training_update
        self.writer.add_scalar("Gate/Avg_Raw_Reward", avg_raw_reward, log_step)
        self.writer.add_scalar("Gate/Execute_Label_Count", n_execute, log_step)
        self.writer.add_scalar("Gate/Reject_Label_Count", n_reject, log_step)
        self.writer.add_scalar("Gate/Avg_Feedback_Weight", avg_feedback_weight, log_step)

        print(
            f"[Gate Train] samples={len(self.memory)}, "
            f"step={current_step}, "
            f"loss={avg_loss:.4f}, "
            f"acc={avg_acc:.3f}, "
            f"avg_raw_reward={avg_raw_reward:.3f}, "
            f"avg_feedback_weight={avg_feedback_weight:.3f}, "
            f"execute={n_execute}, reject={n_reject}"
        )

        for row in epoch_metrics:
            row.update({
                "avg_raw_reward": avg_raw_reward,
                "avg_feedback_weight": avg_feedback_weight,
                "execute_count": int(n_execute),
                "reject_count": int(n_reject),
                "sample_count": len(self.memory),
            })

        file_exists = os.path.isfile(self.gate_training_csv)
        pd.DataFrame(epoch_metrics).to_csv(
            self.gate_training_csv,
            mode="a",
            header=not file_exists,
            index=False
        )

        self.memory.clear()

    def save_model(self, filename):
        """
        Save gate model.
        """
        full_path = os.path.join(self.model_save_dir, filename)
        torch.save(self.model.state_dict(), full_path)
        print(f"[Gate Model] Saved model to {full_path}")

    def save_model_to_path(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[Gate Model] Saved model to {path}")

    def load_model(self, path):
        """
        Load gate model.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"[Gate Model] Loaded model from {path}")

    def record_gate_log(self):
        """
        Save gate training loss/accuracy logs.
        """
        if self.loss_history:
            steps, losses = zip(*self.loss_history)
            loss_csv_path = os.path.join(self.run_dir, "gate_loss_log.csv")
            df_loss = pd.DataFrame({
                "step": steps,
                "loss": losses
            })
            df_loss.to_csv(loss_csv_path, index=False)
            print(f"[Gate Plot] Gate loss data saved to {loss_csv_path}")

        if self.acc_history:
            steps, accs = zip(*self.acc_history)
            acc_csv_path = os.path.join(self.run_dir, "gate_acc_log.csv")
            df_acc = pd.DataFrame({
                "step": steps,
                "accuracy": accs
            })
            df_acc.to_csv(acc_csv_path, index=False)
            print(f"[Gate Plot] Gate accuracy data saved to {acc_csv_path}")

