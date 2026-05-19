"""
Likelihood Ratio Attack (LiRA) from Carlini et al.,
"Membership Inference Attacks From First Principles" (2022).

Approximates the LiRA score without shadow models by using a reference model
as a proxy for the non-member loss distribution. Per-token log-likelihood
ratios are computed, and the score is a Gaussian test statistic:

    score = (μ_target - μ_ref) / sqrt((σ_target² + σ_ref²) / 2 + ε)

This gives a calibrated signal: positive when the target model has higher
per-token loss than the reference, consistent with the forget-set convention
used by other attacks in this codebase.
"""

import numpy as np
from evals.metrics.mia.all_attacks import Attack
from evals.metrics.utils import tokenwise_logprobs


class LiRAAttack(Attack):
    def setup(self, reference_model, output_temperature=1.0, **kwargs):
        self.reference_model = reference_model
        self.output_temperature = output_temperature

    def compute_batch_values(self, batch):
        target_lp = tokenwise_logprobs(self.model, batch, output_temperature=self.output_temperature)
        ref_lp = tokenwise_logprobs(self.reference_model, batch, output_temperature=self.output_temperature)
        return [
            {"target_lp": t, "ref_lp": r}
            for t, r in zip(target_lp, ref_lp)
        ]

    def compute_score(self, sample_stats):
        """Gaussian likelihood ratio score over per-token losses."""
        target_lp = sample_stats["target_lp"]
        ref_lp = sample_stats["ref_lp"]

        if len(target_lp) == 0:
            return 0.0

        target_losses = -target_lp.float().cpu().numpy()
        ref_losses = -ref_lp.float().cpu().numpy()

        mu_t, sigma_t = target_losses.mean(), target_losses.std()
        mu_r, sigma_r = ref_losses.mean(), ref_losses.std()

        pooled_var = (sigma_t ** 2 + sigma_r ** 2) / 2
        return float((mu_t - mu_r) / np.sqrt(pooled_var + 1e-8))
