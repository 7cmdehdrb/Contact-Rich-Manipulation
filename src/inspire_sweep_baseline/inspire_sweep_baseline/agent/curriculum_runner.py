"""Bridge RSL-RL rollout boundaries to Manager-based reward curricula.

RSL-RL updates current_learning_iteration *after* collecting each rollout.
Anchor at learn()'s starting index and count actual rollout steps, using the
runner's configured horizon. This preserves its loop, logging and checkpoint
semantics without copying the PPO loop or using the simulator step counter."""

from rsl_rl.runners import OnPolicyRunner

class LearningIterationEnv:
    """Forward VecEnv calls and publish an iteration before each rollout."""

    def __init__(self, env):
        self.env = env
        self._steps_per_iteration = None
        self._steps = 0
        self._start_iteration = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        # RSL-RL assigns this when init_at_random_ep_len=True.
        self.env.episode_length_buf = value

    def begin(self, iteration, steps_per_iteration):
        if steps_per_iteration <= 0:
            raise ValueError("num_steps_per_env must be positive")
        self._start_iteration = iteration
        self._steps_per_iteration = steps_per_iteration
        self._steps = 0
        self.env.unwrapped.set_learning_iteration(iteration)

    def end(self):
        self._steps_per_iteration = None

    def step(self, actions):
        if self._steps_per_iteration is not None and self._steps % self._steps_per_iteration == 0:
            iteration = self._start_iteration + self._steps // self._steps_per_iteration
            self.env.unwrapped.set_learning_iteration(iteration)
        result = self.env.step(actions)
        if self._steps_per_iteration is not None:
            self._steps += 1
        return result


class CurriculumOnPolicyRunner(OnPolicyRunner):
    """Use checkpoint iteration on load; advance curricula only during learn."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(LearningIterationEnv(env), train_cfg, log_dir=log_dir, device=device)

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self.env.begin(self.current_learning_iteration, self.cfg["num_steps_per_env"])
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self.env.end()

    def load(self, *args, **kwargs):
        infos = super().load(*args, **kwargs)
        # Playback never calls learn(), so it stays at this checkpoint stage.
        self.env.unwrapped.set_learning_iteration(self.current_learning_iteration)
        return infos
