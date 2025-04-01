# pi_optimal/planners/cem_planner.py
from abc import ABC, abstractmethod
from .base_planner import BasePlanner
import numpy as np
from pi_optimal.utils.logger import Logger
import os 

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

import gymnasium as gym

class OnlinePlanner(BasePlanner):
    def __init__(self, 
                 env: gym.Env, 
                 eval_env: gym.Env = None, 
                 log_dir: str=None, 
                 logger: Logger=None, 
                 planner_params: dict={"verbose": 1}, 
                 eval_params: dict={"eval_freq": 5000, "n_eval_episodes": 50, "deterministic": False, "render": False},
                 train_params: dict={"total_timesteps": 300000}):
        self.env = env

        if logger is not None:
            self.logger = logger
        else:
            self.hash_id = np.random.randint(0, 100000)
            self.logger = Logger(f"Online-Planner-{self.hash_id}")

        if eval_env is None:
            self.logger.info("No eval environment provided, will not create eval callback")
            self.eval_callback = None
        else:
            self.logger.info("Creating eval callback")
            if log_dir is not None:
                self.logger.info("Creating eval callback with logging")
                self.eval_callback = EvalCallback(
                    eval_env,
                    best_model_save_path=os.path.join(log_dir, 'best_model'),
                    log_path=log_dir,
                    **eval_params
                )
            else:
                self.logger.info("Creating eval callback without logging")
                self.eval_callback = EvalCallback(
                    eval_env,
                    **eval_params
                )
            

        self.logger.info("Creating PPO model")
        self.model = PPO("MlpPolicy", env, **planner_params)


        self.logger.info("Training model")
        self.model.learn(callback=self.eval_callback, **train_params)

        if log_dir is not None:
            best_model_path = os.path.join(log_dir, 'best_model')
            best_model_file = best_model_path + '.zip'
            if os.path.exists(best_model_file):
                self.logger.info("Loading best model and setting as planner model")
                self.model = PPO.load(best_model_path)
            else:
                self.logger.info("Best model not found. Saving final model as backup.")
                final_model_path = os.path.join(log_dir, 'final_model')
                self.model.save(final_model_path)
                self.model = PPO.load(final_model_path)

    def plan(self, inf_dataset):

        selected_row = inf_dataset.df.iloc[-1]
        self.observation = selected_row[self.env.state_columns].values
        self.observation = self.observation.astype(np.float32)
        action_pred = self.model.predict(self.observation.reshape(1, -1))[0]

        return action_pred
    