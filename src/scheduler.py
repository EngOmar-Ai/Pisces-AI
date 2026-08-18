class WarmupStableDecayLRScheduler:
    def __init__(self, optimizer, mode: str, transition_steps: int, learning_rate: float|int, minimum_ratio: float|int = 0.05):

        if transition_steps <= 0:
            raise ValueError("Transition Steps Must Be Greater Than 0.")

        if learning_rate <= 0:
            raise ValueError("Learning Rate Must Be Greater Than 0.")

        if not 0 < minimum_ratio <= 1:
            raise ValueError("Minimum Ration Must Be Between 0 And 1.")

        self.optimizer = optimizer
        self.mode = mode.upper()

        self.min_learning_rate = learning_rate * minimum_ratio
        self.max_learning_rate = learning_rate

        self.learning_rate = self.min_learning_rate

        self.delta = (self.max_learning_rate - self.min_learning_rate) / transition_steps

        self.steps = 0

    def step(self):

        self.learning_rate = self.get_learning_rate()
        self.set_learning_rate(self.learning_rate)
        self.steps += 1

    def get_learning_rate(self):

        if self.mode == 'W':
            return min(self.learning_rate + self.delta, self.max_learning_rate)
        elif self.mode == 'S':
            return self.learning_rate
        elif self.mode == 'D':
            return max(self.learning_rate - self.delta, self.min_learning_rate)

        raise ValueError('Unknown scheduler mode. Please use one of W (Warm Up), S (Stable), or D (Decay).')

    def set_learning_rate(self, learning_rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def set_mode(self, mode):
        mode = mode.upper()

        if mode not in ('W', 'S', 'D'):
            raise ValueError('Unknown scheduler mode. Please use one of W (Warm Up), S (Stable), or D (Decay).')

        self.mode = mode

    def state_dict(self):
        return {
            "steps": self.steps,
            "learning_rate": self.learning_rate,
        }

    def load_state_dict(self, state_dict):
        self.steps = state_dict["steps"]
        self.learning_rate = state_dict["learning_rate"]

        self.set_learning_rate(self.learning_rate)

if __name__ == "__main__":
    ...