#!/usr/bin/env -S python3

import os
import rclpy
import gymnasium as gym
from gymnasium.wrappers import RecordVideo # Import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
import inspect
import sys

from brian_rl_env.brian_gym_env import BrianGymEnv # Import your custom env


# --- Configuration Constants ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

WALKING_REWARD_THRESHOLD = 5000.0
HUGE_TOTAL_TIMESTEPS = 100_000_000
EVAL_FREQ = 5000
VIDEO_RECORD_FREQ_EPOCHS = 10 # Record a video every 10 evaluation runs

N_ENVS_TRAINING = 8

# Training Hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
N_EPOCHS_PPO = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01

# Define the video save directory
VIDEO_SAVE_DIR = "/root/Documents/brian_ws/src/brian_rl_env/video"
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True) # Ensure the directory exists

# --- Custom Callbacks ---

class CustomEvalAndVideoCallback(EvalCallback):
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq,
                 deterministic=True, render_freq_epochs=10, video_save_folder="", verbose=1):
        super().__init__(eval_env, best_model_save_path=best_model_save_path,
                         log_path=log_path, eval_freq=eval_freq,
                         deterministic=deterministic,
                         verbose=verbose)
        self.render_freq_epochs = render_freq_epochs
        self.current_epoch_count = 0
        self.video_save_folder = video_save_folder # Store the video save folder

        print(f"DEBUG_CB_INIT: Initial eval_env type IN CALLBACK INIT: {type(eval_env)}")
        print(f"DEBUG_CB_INIT: Initial eval_env is Monitor? {isinstance(eval_env, Monitor)}")
        if isinstance(eval_env, Monitor):
            print(f"DEBUG_CB_INIT: Unwrapped type: {type(eval_env.env)}")
            print(f"DEBUG_CB_INIT: Unwrapped is BrianGymEnv? {isinstance(eval_env.env, BrianGymEnv)}")
            # If wrapped by Monitor, check the next layer for RecordVideo
            if isinstance(eval_env.env, RecordVideo):
                print(f"DEBUG_CB_INIT: Unwrapped (Monitor.env) is RecordVideo.")
                print(f"DEBUG_CB_INIT: Unwrapped (Monitor.env.env) type: {type(eval_env.env.env)}")
                print(f"DEBUG_CB_INIT: Unwrapped (Monitor.env.env) is BrianGymEnv? {isinstance(eval_env.env.env, BrianGymEnv)}")


    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        # Check if it's time for a full evaluation based on total timesteps,
        # and if it is, increment our internal epoch counter.
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            self.current_epoch_count += 1
            if self.current_epoch_count % self.render_freq_epochs == 0:
                self.logger.info(f"Triggering video recording for evaluation epoch {self.current_epoch_count} "
                                 f"at {self.num_timesteps} total timesteps.")
                # The video recording is handled by the RecordVideo wrapper itself.
                # We just need to make sure the environment is reset and steps are taken.
                # The EvalCallback will automatically call step/reset on the wrapped env.
                pass # No explicit call needed here because RecordVideo handles it on reset/step

        return continue_training

    # Removed _record_evaluation_video as RecordVideo wrapper handles this.
    # The EvalCallback will automatically trigger reset() on eval_env_for_callback
    # which is wrapped with RecordVideo, starting a new video recording.


def main():
    rclpy.init(args=None)

    print(f"DEBUG_MAIN: sys.executable: {sys.executable}")
    print(f"DEBUG_MAIN: sys.path: {sys.path}")
    print("DEBUG_MAIN: This is the VERIFIED version of train_brian.py with the eval_env_single fix.")

    log_dir = "./brian_ppo_logs"
    os.makedirs(log_dir, exist_ok=True)

    print(f"DEBUG_MAIN: Before make_vec_env. Current type of BrianGymEnv class for make_vec_env: {BrianGymEnv}")

    train_env = make_vec_env(BrianGymEnv, n_envs=N_ENVS_TRAINING, seed=0,
                       monitor_dir=log_dir,
                       wrapper_kwargs=dict(robot_name='brian'))

    print(f"DEBUG_MAIN: train_env type after make_vec_env: {type(train_env)}")

    checkpoint_callback = CheckpointCallback(
        save_freq=EVAL_FREQ,
        save_path=log_dir,
        name_prefix="brian_model"
    )

    print(f"DEBUG_MAIN: About to define eval_env_single. Current type of BrianGymEnv class for eval_env_single: {BrianGymEnv}")

    # --- CRITICAL FIX: Create a SEPARATE, SINGLE environment for evaluation. ---
    # Wrap BrianGymEnv with RecordVideo first, then with Monitor.
    # RecordVideo needs to be directly above BrianGymEnv to access its render() method.
    eval_env_raw = BrianGymEnv(robot_name='brian', render_mode='rgb_array') # Important: Set render_mode for video
    eval_env_wrapped_with_video = RecordVideo(
        eval_env_raw,
        video_folder=VIDEO_SAVE_DIR,
        episode_trigger=lambda x: x % VIDEO_RECORD_FREQ_EPOCHS == 0, # Record every N episodes
        name_prefix="brian_eval_video"
    )
    eval_env_for_callback = Monitor(eval_env_wrapped_with_video) # Monitor the video-wrapped env

    print(f"DEBUG_MAIN: eval_env_for_callback type after definition: {type(eval_env_for_callback)}")

    eval_and_video_callback = CustomEvalAndVideoCallback(eval_env_for_callback,
                                                         best_model_save_path=log_dir,
                                                         log_path=log_dir,
                                                         eval_freq=EVAL_FREQ,
                                                         deterministic=True,
                                                         render_freq_epochs=VIDEO_RECORD_FREQ_EPOCHS,
                                                         video_save_folder=VIDEO_SAVE_DIR # Pass the video folder
                                                        )

    early_stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=WALKING_REWARD_THRESHOLD,
        verbose=1
    )

    eval_and_video_callback.callback_on_eval = early_stop_callback


    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS_PPO,
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./brian_ppo_tensorboard/",
        device="cuda"
    )

    print("Starting training...")
    try:
        model.learn(
            total_timesteps=HUGE_TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_and_video_callback],
            progress_bar=True
        )
        print("Training finished.")
        model.save(os.path.join(log_dir, "brian_final_model"))
        print("Final model saved.")

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        train_env.close()
        eval_env_for_callback.close()
        rclpy.shutdown()

if __name__ == "__main__":
    main()