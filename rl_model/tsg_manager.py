# rl_model/tsg_manager.py
import csv
import os
from pathlib import Path
from rl_model.rl_module import SelfGateAgent


class TSGManager:
    def __init__(self, tsg_mode="off", exp_name="default_run",
                 lr=5e-4, train_interval=32, hidden_dims=(16, 16),
                 predict_model_path=None):
        self.tsg_mode = tsg_mode
        self.train_interval = train_interval
        self.next_save_step = 10000

        self.run_root = Path(os.environ.get(
            "RUN_DIR",
            Path(__file__).resolve().parent / "rl_logs"
        ))
        self.run_root.mkdir(parents=True, exist_ok=True)

        self.train_latest_model_path = self.run_root / "task_self_gate_v2_latest.pt"
        default_predict_model_path = (
                Path(__file__).resolve().parent
                / "saved_models"
                / "task_self_gate_v2_latest_tsg_search_8878185_task7.pt"
        )

        self.predict_model_path = (
            Path(predict_model_path)
            if predict_model_path
            else default_predict_model_path
        )

        if tsg_mode == "audit":
            self.deploy_decision_csv = self.run_root / "task_self_gate_audit_decision_log.csv"
            self.deploy_reward_csv = self.run_root / "task_self_gate_audit_reward_log.csv"
        else:
            self.deploy_decision_csv = self.run_root / "task_self_gate_decision_log.csv"
            self.deploy_reward_csv = self.run_root / "task_self_gate_reward_log.csv"

        self.gate_agent = None
        if tsg_mode in ("train", "predict", "audit"):
            if tsg_mode == 'train':
                load_path = self.train_latest_model_path \
                    if self.train_latest_model_path.exists() else None
            elif tsg_mode in ('predict', 'audit'):
                if not self.predict_model_path.exists():
                    raise FileNotFoundError(
                        f"[TSG] Dispatch model not found: "
                        f"{self.predict_model_path}")
                load_path = self.predict_model_path

            self.gate_agent = SelfGateAgent(
                exp_name=f"SHARED_TSG_{exp_name}",
                model_path=load_path,
                input_dim=6,
                hidden_dims=hidden_dims,
                lr=lr)

    def _append_csv_row(self, path, row):
        file_exists = path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def log_tsg_decision(
        self,
        *,
        decision_step,
        category,
        cand_av_id,
        target_platoon_leader_id,
        score,
        reject_prob,
        execute_prob,
        tsg_execute,
        real_execute,
    ):
        row = {
            "decision_step": decision_step,
            "category": category,
            "cand_av_id": cand_av_id,
            "target_platoon_leader_id": target_platoon_leader_id,
            "score": score,
            "reject_prob": reject_prob,
            "execute_prob": execute_prob,
            "tsg_execute": tsg_execute,
            "real_execute": real_execute,
        }
        self._append_csv_row(self.deploy_decision_csv, row)

    def log_tsg_reward(
        self,
        *,
        decision_step,
        reward_step,
        category,
        cand_av_id,
        target_platoon_leader_id,
        final_reward,
        tsg_execute,
        real_execute,
    ):
        row = {
            "decision_step": decision_step,
            "reward_step": reward_step,
            "category": category,
            "cand_av_id": cand_av_id,
            "target_platoon_leader_id": target_platoon_leader_id,
            "final_reward": final_reward,
            "tsg_execute": tsg_execute,
            "real_execute": real_execute,
        }
        self._append_csv_row(self.deploy_reward_csv, row)

    def train_if_needed(self, step, st):
        if self.tsg_mode != "train":
            return
        if self.gate_agent is None:
            raise ValueError("[TSG] tsg_mode='train' requires gate_agent")

        if len(self.gate_agent.memory) < self.train_interval:
            return

        self.gate_agent.train_on_recorded(
            current_step=step,
            epochs=5,
            batch_size=max(1, int(self.train_interval / 2))
        )

        if step > self.next_save_step or step == st * 10 - 1:
            self.gate_agent.save_model_to_path(
                self.run_root / f"task_self_gate_v2_step{step}.pt"
            )
            self.save_latest()
            self.next_save_step += 30000

    def save_latest(self):
        if self.gate_agent is None:
            return
        self.gate_agent.save_model_to_path(self.train_latest_model_path)
