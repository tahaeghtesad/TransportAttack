from stable_baselines3.common.callbacks import BaseCallback
import torch.utils.tensorboard as tb
from stable_baselines3.common.vec_env import VecEnv


class EnvLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(EnvLoggerCallback, self).__init__(verbose)
        # self.env = self.training_env  # VecEnv[BasicDefenderEnv]
        self.done_steps = []

    def _on_step(self) -> bool:
        env: VecEnv = self.locals['env']
        detected = env.get_attr('detected')
        step_counts = env.get_attr('step_count')
        # detected = env.unwrapped.detected
        # step_counts = env.unwrapped.step_count

        for i, (d, s) in enumerate(zip(detected, step_counts)):
            if d:
                self.done_steps.append(s)

        return True

    def _on_rollout_end(self) -> None:

        self.logger.record('env/mean_detection_step', sum(self.done_steps) / (len(self.done_steps) + 1e-8))
        self.done_steps = []
