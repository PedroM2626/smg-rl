"""Reward for the Luigi's Purple Coins missions: coins, damage and a one-time approach to the star.

Settings (the `purple_coin_reward` config block):
- `coin_reward`: paid per purple coin gained (the owner chose +1).
- `threshold`: coin count (100) at which the approach credit switches on.
- `approach_total`: the whole approach credit, paid as the distance to the star shrinks below its previous best;
  it never pays for moving away or for regaining lost distance, so it cannot exceed this total.
- `max_gain_per_decision`: a larger jump of the counter is treated as a memory fault, not as coins.
Damage (-1 per point) and the star (+50) use the ordinary `rewards.damage` and `rewards.completion` weights, and
death carries no penalty. The star position is the owner's `star_ready` state: Mario stands one jump below the star.
"""
import numpy as np

KEYS = ('coin_reward', 'threshold', 'approach_total', 'max_gain_per_decision')


def validate(settings):
    values = [settings.get(k) for k in KEYS]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in values):
        raise ValueError('The purple coin reward needs finite numbers for every setting.')
    coin, threshold, approach, maximum = values
    if coin <= 0 or approach < 0 or threshold < 1 or maximum < 1 or threshold != int(threshold):
        raise ValueError('Purple coin reward values are out of range.')


class PurpleCoinReward:
    """Pays coin gains and, once the threshold is reached, new best distances to the star."""

    def __init__(self, settings, star):
        validate(settings)
        self.coin, self.threshold = float(settings['coin_reward']), int(settings['threshold'])
        self.approach_total, self.maximum = float(settings['approach_total']), int(settings['max_gain_per_decision'])
        self.star = np.asarray(star, dtype=float)
        self.reset(0, None)

    def reset(self, count, position):
        self.count, self.best_count = int(count), int(count)
        self.armed, self.span, self.best_distance = False, None, None
        if position is not None:
            self._arm(position)

    def distance(self, position):
        return float(np.linalg.norm(np.asarray(position, dtype=float) - self.star))

    def _arm(self, position):
        if not self.armed and self.count >= self.threshold:
            self.armed = True
            self.span = self.best_distance = self.distance(position)

    def update(self, count, position):
        """Return (coins gained, coin reward, approach reward) for one decision."""
        count = int(count)
        gained = count - self.count
        if gained > self.maximum:
            raise ValueError(f'The purple coin counter jumped by {gained} in one decision.')
        gained = max(0, gained)  # A drop is a respawn (the counter returns to zero) and is never paid or punished.
        self.count = count
        self.best_count = max(self.best_count, count)
        approach = 0.0
        if self.armed and self.span is not None and self.span >= 1.0:
            distance = self.distance(position)
            if distance < self.best_distance:
                approach = self.approach_total * (self.best_distance - distance) / self.span
                self.best_distance = distance
        elif not self.armed:
            self._arm(position)  # The credit starts on the decision after the threshold is reached.
        return gained, self.coin * gained, approach
