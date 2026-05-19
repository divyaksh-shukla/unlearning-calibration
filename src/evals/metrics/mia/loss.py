"""
Straight-forward LOSS attack, as described in https://ieeexplore.ieee.org/abstract/document/8429311
"""

from evals.metrics.mia.all_attacks import Attack
from evals.metrics.utils import evaluate_probability


class LOSSAttack(Attack):
    def setup(self, output_temperature=1.0, **kwargs):
        self.output_temperature = output_temperature
    
    def compute_batch_values(self, batch):
        """Compute probabilities and losses for the batch."""
        return evaluate_probability(self.model, batch, output_temperature=self.output_temperature)

    def compute_score(self, sample_stats):
        """Return the average loss for the sample."""
        return sample_stats["avg_loss"]
