from torch.optim import Optimizer
import sys

class WarmupStableDecayLRScheduler:
    """
    Learning rate scheduler implementing the Warmup-Stable-Decay (WSD) schedule.

    The schedule has three modes, switched manually via `set_mode`:
        - 'W' (Warmup): linearly increases the learning rate from
          `min_learning_rate` to `max_learning_rate` over `transition_steps` steps.
        - 'S' (Stable): holds the learning rate constant.
        - 'D' (Decay): linearly decreases the learning rate from
          `max_learning_rate` down to `min_learning_rate` over `transition_steps` steps.

    The scheduler starts in warmup mode with the learning rate initialized to
    `min_learning_rate`, and `step()` must be called once per training step to
    update the optimizer's learning rate.
    """

    def __init__(self, optimizer : Optimizer, mode: str, transition_steps: int, learning_rate: float | int, minimum_ratio: float | int = 0.05) -> None:
        """
        Initialize the scheduler and attach it to an optimizer.

        Args:
            optimizer: A PyTorch-style optimizer exposing a `param_groups` list,
                whose learning rate will be updated by this scheduler.
            mode: Initial schedule mode. Case-insensitive; must resolve to one of
                'W' (Warmup), 'S' (Stable), or 'D' (Decay).
            transition_steps: Number of `step()` calls needed to move from
                `min_learning_rate` to `max_learning_rate` (or vice versa) during
                the warmup/decay phases. Must be greater than 0.
            learning_rate: The peak (maximum) learning rate. Must be greater than 0.
            minimum_ratio: Fraction of `learning_rate` used as the floor
                (`min_learning_rate = learning_rate * minimum_ratio`). Must be in
                the range [0, 1]. Defaults to 0.05.

        Raises:
            SystemExit: If `transition_steps`, `learning_rate`, or `minimum_ratio`
                fail validation. Prints a report of which checks failed before exiting.
        """

        if transition_steps <= 0 or learning_rate <= 0 or not 0 < minimum_ratio <= 1:
            sys.exit(f"Invalid Arguments, Please Check The Argument Report For More Details\n\n--------------- Arguments Report ---------------\n\nTransitions Steps Greater Than 0: {transition_steps > 0}\nLearning Rate Greater Than 0: {learning_rate > 0}\nMinimum Ratio Between 0 And 1: {1 >= minimum_ratio >= 0}")

        self.optimizer = optimizer
        self.mode = mode.upper()

        self.min_learning_rate = learning_rate * minimum_ratio
        self.max_learning_rate = learning_rate

        self.learning_rate = self.min_learning_rate

        self.delta = (self.max_learning_rate - self.min_learning_rate) / transition_steps

        self.steps = 0

    def step(self) -> None:
        """
        Advance the scheduler by one step and update the optimizer's learning rate.

        Behavior depends on the current `mode`:
            - 'W': increases the learning rate by `delta`, clamped to `max_learning_rate`.
            - 'D': decreases the learning rate by `delta`, clamped to `min_learning_rate`.
            - Any other mode (e.g. 'S'): leaves the learning rate unchanged.

        Applies the updated learning rate via `set_learning_rate` and increments
        the internal step counter.
        """

        if self.mode == 'W':
            updated_learning_rate = min(self.learning_rate + self.delta, self.max_learning_rate)
        elif self.mode == 'D':
            updated_learning_rate = max(self.learning_rate - self.delta, self.min_learning_rate)
        else:
            updated_learning_rate = self.learning_rate

        self.set_learning_rate(updated_learning_rate)

        self.steps += 1

    def set_learning_rate(self, learning_rate: float|int) -> None:
        """
        Directly set the learning rate on the scheduler and the optimizer.

        Args:
           learning_rate: The new learning rate to apply. Updates `self.learning_rate`
               and every `param_group['lr']` in `self.optimizer.param_groups`.
        """

        self.learning_rate = learning_rate

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def set_mode(self, mode: str) -> None:
        """
        Change the scheduler's current mode.

        Args:
            mode: The new mode. Whitespace is stripped and casing is normalized;
                must resolve to one of 'W' (Warm Up), 'S' (Stable), or 'D' (Decay).

        Raises:
            ValueError: If `mode` does not resolve to 'W', 'S', or 'D'.
        """

        mode = mode.strip().upper()

        if mode not in ('W', 'S', 'D'):
            raise ValueError('Unknown scheduler mode. Please use one of W (Warm Up), S (Stable), or D (Decay).')

        self.mode = mode

    def state_dict(self) -> dict:
        """
        Return a dict capturing the scheduler's current state for checkpointing.

        Returns:
            A dict with keys "learning_rate", "steps", and "mode", reflecting the
            scheduler's current values.
        """

        return {
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "mode": self.mode,
        }

    def load_state_dict(self, state_dict:dict) -> None:
        """
        Restore the scheduler's state from a dict produced by `state_dict()`.

        Args:
            state_dict: A dict containing "steps", "learning_rate", and "mode" keys,
                as returned by `state_dict()`. Restores `self.steps` directly, and
                applies "learning_rate" and "mode" via `set_learning_rate` and
                `set_mode` respectively (so the optimizer's `param_groups` are
                updated and the mode is validated).
        """

        self.steps = state_dict["steps"]

        self.set_learning_rate(state_dict["learning_rate"])
        self.set_mode(state_dict["mode"])

if __name__ == "__main__":
    ...