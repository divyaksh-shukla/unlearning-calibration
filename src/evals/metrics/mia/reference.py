"""
Reference-based attacks.
"""

from evals.metrics.mia.all_attacks import Attack
from evals.metrics.utils import evaluate_probability


class ReferenceAttack(Attack):
    def setup(self, reference_model, output_temperature=1.0, **kwargs):
        """Setup reference model."""
        self.reference_model = reference_model
        self.output_temperature = output_temperature

    def compute_batch_values(self, batch):
        """Compute loss scores for both target and reference models."""
        ref_results = evaluate_probability(self.reference_model, batch, output_temperature=self.output_temperature)
        target_results = evaluate_probability(self.model, batch, output_temperature=self.output_temperature)
        return [
            {"target_loss": t["avg_loss"], "ref_loss": r["avg_loss"]}
            for t, r in zip(target_results, ref_results)
        ]

    def compute_score(self, sample_stats):
        """Score using difference between target and reference model losses."""
        return sample_stats["target_loss"] - sample_stats["ref_loss"]
