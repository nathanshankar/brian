import os
import rclpy
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import numpy as np

from quadruped_env import QuadrupedEnv


# Define paths
URDF_PATH = "/home/nathan/brian/src/brian_description/urdf/brian.urdf"
LOG_DIR = "./ppo_quadruped_logs/"
MODEL_DIR = "./ppo_quadruped_models/"
VIDEO_DIR = "/home/nathan/brian/src/brian_pybullet/video/"


class VideoRecorderCallback(BaseCallback):
    """
    A custom callback to record videos of the agent's behavior during training.
    """
    def __init__(self, eval_env_fn, video_freq: int, render_steps: int = 250, video_dir: str = "./videos/", verbose: int = 0):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.video_freq = video_freq
        self.render_steps = render_steps
        self.video_dir = video_dir
        os.makedirs(self.video_dir, exist_ok=True)
        self.recorded_videos_count = 0

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout.
        """
        # Save a video every `video_freq` rollouts
        if self.n_calls % self.video_freq == 0:
            print(f"--- Recording video at iteration {self.n_calls} (total timesteps: {self.num_timesteps}) ---")
            
            render_env = self.eval_env_fn()
            
            frames = []
            obs, info = render_env.reset()
            
            for _ in range(self.render_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = render_env.step(action)
                
                frame = render_env.render()
                if frame is not None:
                    frames.append(frame)
                
                if terminated or truncated:
                    obs, info = render_env.reset()

            render_env.close()

            if frames:
                video_filename = os.path.join(self.video_dir, f"quadruped_iter_{self.n_calls}_ts_{self.num_timesteps}.mp4")
                try:
                    imageio.mimsave(video_filename, frames, fps=render_env.metadata['render_fps'])
                    print(f"Video saved to: {video_filename}")
                    self.recorded_videos_count += 1
                except Exception as e:
                    print(f"Error saving video {video_filename}: {e}")
            else:
                print("No frames captured for video.")

    def _on_step(self) -> bool:
        """
        Method called at each step of the training loop.
        Must return True to continue training, False to stop.
        """
        return True # Required by BaseCallback


def train_quadruped_agent(timesteps=1000000, log_dir=LOG_DIR, model_dir=MODEL_DIR, video_dir=VIDEO_DIR):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    if not rclpy.ok():
        rclpy.init(args=None)

    try:
        env = make_vec_env(QuadrupedEnv, env_kwargs={'urdf_path': URDF_PATH, 'render_mode': None}, n_envs=1)

        def create_render_env():
            return QuadrupedEnv(urdf_path=URDF_PATH, robot_name="brian", render_mode='rgb_array')

        callbacks = [
            CheckpointCallback(
                save_freq=10000,
                save_path=model_dir,
                name_prefix="quadruped_ppo_model",
                save_replay_buffer=True,
                save_vecnormalize=True
            ),
            VideoRecorderCallback(
                eval_env_fn=create_render_env,
                video_freq=1, # Changed from 5 to 1 to save video every iteration/rollout
                render_steps=250,
                video_dir=video_dir
            )
        ]

        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

        print(f"Starting training for {timesteps} timesteps...")
        model.learn(total_timesteps=timesteps, callback=callbacks)
        print("Training finished.")

        final_model_path = os.path.join(model_dir, "final_quadruped_ppo_model")
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}.zip")

    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    train_quadruped_agent(timesteps=500000, log_dir=LOG_DIR, model_dir=MODEL_DIR, video_dir=VIDEO_DIR)
