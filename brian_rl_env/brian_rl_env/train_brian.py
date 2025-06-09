# train_brian.py (MODIFIED)
#!/usr/bin/env -S python3

import rclpy
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from brian_rl_env.brian_gym_env import BrianGymEnv # Import your custom env

import os

def main():
    # Initialize ROS 2 context ONCE at the start of the training script
    rclpy.init(args=None) 

    log_dir = "./brian_ppo_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create the training environment
    # Note: `render_mode='human'` here enables the PyBullet GUI for the training environment.
    env = make_vec_env(BrianGymEnv, n_envs=1, seed=0,
                       monitor_dir=log_dir,
                       wrapper_kwargs=dict(robot_name='brian', render_mode='human'))

    # Callbacks for saving best model and checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, save_path=log_dir, name_prefix="brian_model"
    )

    # Create the evaluation environment (no rendering for evaluation for performance)
    eval_env = BrianGymEnv(robot_name='brian') 
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=50000,
                                 deterministic=True) # REMOVED render_freq

    # Initialize the PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./brian_ppo_tensorboard/",
        device="auto"
    )

    # Train the agent
    print("Starting training...")
    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        print("Training finished.")
        # Save the final model
        model.save(os.path.join(log_dir, "brian_final_model"))
        print("Final model saved.")

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        # Clean up
        env.close()
        eval_env.close()
        rclpy.shutdown() 

if __name__ == "__main__":
    main()