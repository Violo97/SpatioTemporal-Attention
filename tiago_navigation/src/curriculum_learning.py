#!/usr/bin/env python
import os
import numpy as np
import rospy
import rospkg
import gymnasium as gym
from datetime import datetime
import torch
import torch.nn as nn

from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

# Assicura la registrazione dell'env
from openai_ros.task_envs.tiago import tiago_navigation  # noqa

# La tua rete
from rl_algorithms.SpatioTemporal import SpatioTemporalBase, network_config


# ---------------------- Feature Extractor ----------------------
class TiagoSB3Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int , attention_method: str ):
        
        super().__init__(observation_space, features_dim)
        if attention_method == "Spatio_Temporal" or attention_method == "Depth_Temporal":
            base_out = 2 * network_config["spatial_value_mlp_layers"][-1]
        else:
            base_out = network_config["spatial_value_mlp_layers"][-1]
        
        self.backbone = SpatioTemporalBase()
        self.proj = nn.Identity() if features_dim == base_out else nn.Sequential(
            nn.Linear(base_out, features_dim), nn.ReLU()
        )

    def forward(self, obs_dict):
        lidar = obs_dict["cartesian_scan"].float()
        wpts  = obs_dict["waypoints"].float()
        tagd  = obs_dict["tagd"].float()
        feats = self.backbone(lidar, wpts, tagd)
        return self.proj(feats)

# ---------------------- Env factory + Monitor ----------------------

# Wrapper to expose Gymnasium's terminated/truncated to Monitor
class EpisodeEndFlags(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info = dict(info)  # ensure writable
            info["terminated"] = bool(terminated)
            info["truncated"] = bool(truncated)
            # If your env already sets info["is_success"], Monitor will log it (see below).
            # Otherwise compute/set it here based on your success condition.
            # info["is_success"] = <your_bool>
        return obs, reward, terminated, truncated, info


def make_test_env():
    env = gym.make("TiagoNavigation-v0")
    env = EpisodeEndFlags(env)  # expose end flags for logging
    rospack = rospkg.RosPack()
    test_dir = os.path.join(rospack.get_path("tiago_navigation"), "test_results/eval_monitor")
    os.makedirs(test_dir, exist_ok=True)
    # Ask Monitor to record our custom keys too:
    env = Monitor(env, filename=test_dir,
                  info_keywords=("is_success", "terminated", "truncated"))
    return env


def make_env():
    env = gym.make("TiagoNavigation-v0")
    # Log monitor (reward/length) su file
    rospack = rospkg.RosPack()
    log_dir = os.path.join(rospack.get_path("tiago_navigation"), "train_results/monitor")
    fname = os.path.join(log_dir, f"monitor_{datetime.now():%Y%m%d-%H%M%S}")
    env = Monitor(env, filename=fname)
    return env


def build_model(algo: str, venv, tb_log: str , save_dir: str, policy_kwargs: dict , load_model = False):
    """
    Iperparametri consigliati per ambienti continui lenti (Gazebo/ROS),
    con action space: linear [0.0, 0.6], angular [-0.5, 0.5].
    """

    # --- calcolo rumore per-dimensione in base ai bounds dell'env ---
    low  = venv.action_space.low.astype(np.float32)
    high = venv.action_space.high.astype(np.float32)
    rng  = high - low
    n_actions = int(venv.action_space.shape[0])

    # media del rumore: spingi leggermente in avanti la componente lineare
    # [linear_mean, angular_mean]
    noise_mean = np.array([
        0.10,   # m/s (piccolo bias in avanti per l’esplorazione)
        0.00    # rad/s
    ], dtype=np.float32)

    # deviazioni standard del rumore: ~10% dell’intervallo
    noise_sigma = 0.10 * rng  # per es.: [0.06, 0.10]

    # --- SAC: robusto, stabile, entropia automatica ---
    if algo == "SAC":
        if load_model:
            model = SAC.load(os.path.join(save_dir, "SAC_final.zip") , env=venv, print_system_info=False)
            if os.path.exists(os.path.join(save_dir, "SAC_replaybuffer.pkl")):
                model.load_replay_buffer(os.path.join(save_dir, "SAC_replaybuffer.pkl"))
        else:
            model = SAC(
                policy="MultiInputPolicy",
                env=venv,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,        # un po' più veloce
                buffer_size=100_000,       # meno grande per compito semplice
                batch_size=256,
                tau=0.01,                  # update un po' più rapido
                gamma=0.95,                # orizzonte più corto
                train_freq=64,
                gradient_steps=64,
                ent_coef=0.05,             # fisso (meno esplorazione random)
                verbose=1,
                learning_starts=1000,
            )
        return model

    # --- TD3: twin critics, policy delay e target policy smoothing ---
    if algo == "TD3":
        from stable_baselines3.common.noise import NormalActionNoise
        action_noise = NormalActionNoise(
            mean=noise_mean,         # bias in avanti sul lineare
            sigma=noise_sigma        # 10% dell’intervallo per dimensione
        )

        # target policy noise & clip proporzionali all'intervallo azione
        target_policy_noise = 0.20 * rng     # smoothing sul target
        target_noise_clip   = 0.50 * rng     # clip del rumore target
        if load_model:
            model = TD3.load(os.path.join(save_dir, "TD3_final.zip") , env=venv, print_system_info=False)
            if os.path.exists(os.path.join(save_dir, "TD3_replaybuffer.pkl")):
                model.load_replay_buffer(os.path.join(save_dir, "TD3_replaybuffer.pkl"))
        else:
            model = TD3(
                policy="MultiInputPolicy",
                env=venv,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,     
                buffer_size=300_000,
                batch_size=256,
                tau=0.005,               
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                action_noise=action_noise,
                target_policy_noise=float(np.mean(target_policy_noise)),  # SB3 accetta float
                target_noise_clip=float(np.mean(target_noise_clip)),      # usa media tra le due dims
                policy_delay=2,          # default TD3
                verbose=1,
                #tensorboard_log=tb_log,
                learning_starts=5000,
            )
        return model

    
    if algo == "DDPG":
        from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
        ou_sigma = 0.20 * rng       
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=noise_mean,
            sigma=ou_sigma,
            theta=0.15, dt=1.0
        )
        if load_model:
            model = DDPG.load(os.path.join(save_dir, "DDPG_final.zip") , env=venv, print_system_info=False)
            if os.path.exists(os.path.join(save_dir, "DDPG_replaybuffer.pkl")):
                model.load_replay_buffer(os.path.join(save_dir, "DDPG_replaybuffer.pkl"))
                rospy.loginfo("Replay buffer caricato.")
        else:
            model = DDPG(
                policy="MultiInputPolicy",
                env=venv,
                policy_kwargs=policy_kwargs,
                learning_rate=1e-4,
                buffer_size=300_000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                action_noise=action_noise,
                verbose=1,
                #tensorboard_log=tb_log,
                learning_starts=5000, 
            )
        return model

    raise ValueError(f"Unknown algorithm: {algo} (use sac, td3, ddpg)")

def evaluate_with_success(model, env, n_eval_episodes=2, deterministic=True):
    successes = []

    def cb(locals_, globals_):
        dones = locals_["dones"]
        infos = locals_["infos"]

        # VecEnv vs singolo env
        if isinstance(dones, (list, tuple, np.ndarray)):
            for d, info in zip(dones, infos):
                if d and info is not None and "is_success" in info:
                    successes.append(bool(info["is_success"]))
        else:
            if dones and infos is not None and "is_success" in infos:
                successes.append(bool(infos["is_success"]))
        return True

    mean_rew, std_rew = evaluate_policy(
        model, env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        callback=cb,
    )
    sr = float(np.mean(successes)) if successes else 0.0
    return mean_rew, std_rew, sr


def test_phase(model , env , path , algorithm , curr_best_sr = 0.0 , update_lr = 0.6):
    mean_r, std_r, sr = evaluate_with_success(model.policy, env, n_eval_episodes=20)
    rospy.loginfo(f"Mean reward: {mean_r:.2f} +/- {std_r:.2f} | Success rate: {sr*100:.1f}%")
    rospy.loginfo(f"SR_EVAL {sr:.4f}")
    rospy.loginfo(f"Current best success rate: {curr_best_sr:.1f}")
    if sr > curr_best_sr:
        model.save(path + "/" + algorithm + "_best")
        rospy.loginfo("Modello salvato.")
        #save replay buffer
        try:
            model.save_replay_buffer(path + "/" + algorithm + "_replaybuffer_best")
            rospy.loginfo("Replay buffer salvato.")
        except Exception:
            rospy.logwarn("Replay buffer non salvato. L'algoritmo o il vecenv potrebbe non supportarlo.")
        


def training_phase(model , algorithm , path):
    rospy.loginfo("Fase di training iniziata.")
    model.learn(total_timesteps=2500, reset_num_timesteps=False)
    rospy.loginfo("Fase di training completata.")
    #save checpoint finale
    model.save(path + "/" + algorithm + "_final")
    rospy.loginfo("Modello salvato.")
    #save replay buffer
    try:
        model.save_replay_buffer(path + "/" + algorithm + "_replaybuffer")
        rospy.loginfo("Replay buffer salvato.")
    except Exception:
        rospy.logwarn("Replay buffer non salvato. L'algoritmo o il vecenv potrebbe non supportarlo.")

    # save policy of model
    #policy = model.policy
    #policy.save(path + algorithm + "_policy")
    #rospy.loginfo("Policy salvata.")

def set_lr(model, algo , new_lr: float):
    if algo == "SAC":  
        for param_group in model.policy.actor.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Update critic (Q-networks) learning rate  
        for param_group in model.policy.critic.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Update entropy coefficient (alpha) learning rate if using automatic entropy tuning
        if model.ent_coef_optimizer is not None:
            for param_group in model.ent_coef_optimizer.param_groups:
                param_group['lr'] = new_lr

    elif algo == "DDPG":
        for param_group in model.actor.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Update critic optimizer
        for param_group in model.critic.optimizer.param_groups:
            param_group['lr'] = new_lr  
        
        model.learning_rate = new_lr
    
    elif algo == "TD3":
        for param_group in model.actor.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Update critic optimizer (twin Q-networks)
        for param_group in model.critic.optimizer.param_groups:
            param_group['lr'] = new_lr  
        
        model.learning_rate = new_lr
    
    else:
        raise ValueError(f"Unknown algorithm: {algo} (use sac, td3, ddpg)")
    
    return model

if __name__ == "__main__":
    rospy.init_node("tiago_sb3_train", anonymous=True, log_level=rospy.INFO)
    # ROS parameters for training
    algo = rospy.get_param("/Training/algorithm", "sac")              # "sac" | "td3" | "ddpg"
    attention_method = rospy.get_param("/Training/attention_method") # "Spatio_Temporal" | "Spatial" | "Temporal" | "Depth_Temporal" | "Depth"

    if attention_method == "Spatio_Temporal" or attention_method == "Depth_Temporal":
        features_dim = 60
    else:
        features_dim = 30

    test_flag = rospy.get_param("/Training/test", False)    # VecEnv (una sola istanza per ROS)
    #test_flag = True
    if test_flag:
        venv = DummyVecEnv([make_test_env])
    else:
        venv = DummyVecEnv([make_env])
    # Policy/extractor
    policy_kwargs = dict(
        features_extractor_class=TiagoSB3Extractor,
        features_extractor_kwargs=dict(features_dim=features_dim , attention_method=attention_method),
        net_arch=dict(
            pi=[128, 64 , 64],
            qf=[128, 64 , 64],
        ),
    )

    rospack  = rospkg.RosPack()
    base_dir = rospack.get_path("tiago_navigation")
    rospy.loginfo(str(base_dir))
    #tb_log   = os.path.join(base_dir, "tb_logs")
    tb_log = os.path.join(rospack.get_path("tiago_navigation"), "train_results/log")
    save_dir = os.path.join(base_dir, "sb3_checkpoints")
    
    if os.path.exists(os.path.join(save_dir, algo + "_final.zip")):
            model = build_model(algo, venv, tb_log , save_dir , policy_kwargs , load_model = True)
            rospy.loginfo("Checkpoint e replay buffer trovati. Continuo il training dal modello caricato.")
    else:
        model = build_model(algo, venv, tb_log , save_dir, policy_kwargs , load_model = False)
        rospy.loginfo("Nessun checkpoint trovato. Inizio il training da zero.")

    run_dir = os.path.join(tb_log, f"{algo}_{datetime.now():%Y%m%d-%H%M%S}")
    new_logger = configure(run_dir, ["stdout", "tensorboard", "csv"])  # creates progress.csv
    model.set_logger(new_logger)
    old_lr = model.actor.optimizer.param_groups[0]['lr']
    rospy.loginfo(str(old_lr))
    # Method 1: Update learning rate for all networks
    #new_lr = 1e-5

    
    #model = set_lr(model, algo , new_lr)

    if test_flag:
        #value for control if the new policy is better than the other or if needed to update lr
        update_lr = rospy.get_param("/Training/learning_rate_update")
        curr_best_sr = float(rospy.get_param("/Training/curr_sr", 0.0))
        test_phase(model , venv , save_dir , algo , curr_best_sr , update_lr)
    else:
        #load checkpoint and replay buffer if exist and continue training
        training_phase(model , algo , save_dir)