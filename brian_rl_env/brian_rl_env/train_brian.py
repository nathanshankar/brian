# train_brian.py (MODIFIED - Corrected CustomEvalAndVideoCallback)
#!/usr/bin/env -S python3

import os
import rclpy
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from brian_rl_env.brian_gym_env import BrianGymEnv # Import your custom env


class CustomEvalAndVideoCallback(EvalCallback):
    def __init__(self, eval_env, best_model_save_path, log_path, eval_freq,
                 deterministic=True, render_freq_epochs=100, verbose=1):
        super().__init__(eval_env, best_model_save_path=best_model_save_path,
                         log_path=log_path, eval_freq=eval_freq,
                         deterministic=deterministic,
                         # REMOVED: render_freq=1, # This argument is not valid for EvalCallback.__init__
                         verbose=verbose)
        self.render_freq_epochs = render_freq_epochs
        self.current_epoch_count = 0 # Track epochs to trigger video saving

    def _on_step(self) -> bool:
        """
        This method is called by the Stable Baselines3 agent after each training step.
        """
        # Call the parent _on_step to handle standard evaluation logic
        continue_training = super()._on_step()

        # Check if it's time for a full evaluation and video recording
        # The `eval_freq` in EvalCallback refers to `total_timesteps`
        # We need to link this to the concept of "epochs" or number of evaluation runs.
        # Let's use `self.num_timesteps` from the parent class and the `eval_freq`.
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.current_epoch_count += 1
            if self.current_epoch_count % self.render_freq_epochs == 0:
                self._record_evaluation_video()

        return continue_training

    def _record_evaluation_video(self):
        self.logger.info(f"Epoch {self.current_epoch_count}: Starting video recording for evaluation...")

        # Cast eval_env to BrianGymEnv (assuming n_envs=1 for eval_env)
        # Note: If you use make_vec_env for eval, you'll need to access envs[0]
        # For simplicity, ensure eval_env is a single BrianGymEnv instance
        if isinstance(self.eval_env, Monitor): # Unwrap Monitor if present
            env_to_record = self.eval_env.env # Access the underlying env
        else:
            env_to_record = self.eval_env

        if not isinstance(env_to_record, BrianGymEnv):
            self.logger.error("Evaluation environment is not BrianGymEnv. Cannot record video.")
            return

        # Start logging in the PyBullet node via the service
        # The log file path will be printed by the brian_pybullet_node itself
        log_file_path = env_to_record._start_logging() 
        if log_file_path is None:
            self.logger.error("Failed to start logging, skipping video recording.")
            return
        
        # Give some time for the logging to capture an actual episode or a few steps
        # The EvalCallback will run a few episodes as part of its _on_step logic.
        # We assume the logging will capture this.
        # A more robust solution might involve:
        # 1. Resetting the environment after starting logging.
        # 2. Running a fixed number of steps (e.g., 500 steps) in the evaluation environment.
        # 3. Then stopping logging.
        # For now, we are piggybacking on the EvalCallback's default evaluation run.
        # So, the duration of the logged video corresponds to the evaluation episodes.

        # Wait a moment for PyBullet to start capturing frames, if necessary
        # time.sleep(0.1) # Optional, can add a small delay if the start of log is too fast

        # Stop logging
        env_to_record._stop_logging()
        self.logger.info(f"Video logging complete for epoch {self.current_epoch_count}. Saved to: {log_file_path}")
        self.logger.info("To view, open PyBullet GUI and use: `pb.createVisualizer(pb.ER_BULLET_GUI); pb.configureDebugVisualizer(pb.COV_ENABLE_GUI,0); pb.startStateLogging(pb.STATE_LOGGING_VIDEO_PLAYBACK, 'YOUR_LOG_FILE.bullet')`")


def main():
    rclpy.init(args=None) 

    log_dir = "./brian_ppo_logs"
    os.makedirs(log_dir, exist_ok=True)

    # For training, use n_envs > 1 for speed (e.g., 4 or 8 if you have cores)
    # No render_mode for headless training envs
    env = make_vec_env(BrianGymEnv, n_envs=4, seed=0, # <--- Changed n_envs to 4 for speed
                       monitor_dir=log_dir,
                       wrapper_kwargs=dict(robot_name='brian')) 

    checkpoint_callback = CheckpointCallback(
        save_freq=100000, save_path=log_dir, name_prefix="brian_model"
    )

    # Eval env should be a single instance, and it should be wrapped with Monitor
    # No render_mode for eval env by default, as we're doing state logging, not direct rendering.
    eval_env = Monitor(BrianGymEnv(robot_name='brian')) 
    
    # Use the custom callback
    eval_and_video_callback = CustomEvalAndVideoCallback(eval_env,
                                                         best_model_save_path=log_dir,
                                                         log_path=log_dir,
                                                         eval_freq=50000, # Eval every 50k timesteps
                                                         deterministic=True,
                                                         render_freq_epochs=100 # <--- Set to 100 for your requirement
                                                        )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048, # Consider increasing slightly for more data per update (e.g., 4096)
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log="./brian_ppo_tensorboard/",
        device="cuda" # <--- IMPORTANT: Changed to "cuda"
    )

    print("Starting training...")
    try:
        model.learn(
            total_timesteps=5_000_000,
            callback=[checkpoint_callback, eval_and_video_callback], # Use the custom callback
            progress_bar=True
        )
        print("Training finished.")
        model.save(os.path.join(log_dir, "brian_final_model"))
        print("Final model saved.")

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()
        eval_env.close()
        rclpy.shutdown()

if __name__ == "__main__":
    main()