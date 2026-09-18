import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from platoon_split_env_sumo import PlatoonSplitEnvSUMO
from state_builder import StateBuilder
import traci

# === Start SUMO simulation (update config path) ===
sumoBinary = "sumo"  # Use "sumo-gui" for GUI visualization
sumoConfig = "your_sumo_config.sumocfg"  # Replace with your actual SUMO config file
traci.start([sumoBinary, "-c", sumoConfig])

# === Create state builder ===
state_builder = StateBuilder(traci)

# === Define fixed platoon and candidate AVs (update to match your scenario) ===
platoon_ids = ["veh1", "veh2", "veh3"]
candidate_av_ids = ["av1", "av2"]

# === Create SUMO-based environment ===
env = PlatoonSplitEnvSUMO(
    traci=traci,
    state_builder=state_builder,
    platoon_ids=platoon_ids,
    candidate_av_ids=candidate_av_ids,
    threshold=0.5  # Score threshold to decide insertion
)

# === Wrap in vectorized environment for Stable-Baselines3 ===
vec_env = make_vec_env(lambda: env, n_envs=1)

# === Define PPO model ===
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

# === Train the model ===
model.learn(total_timesteps=200_000)

# === Save the trained model ===
os.makedirs("../rl_model", exist_ok=True)
model.save("rl_model/scoring_agent_sumo")

# === Close SUMO connection ===
traci.close()
print("Training complete and model saved.")
