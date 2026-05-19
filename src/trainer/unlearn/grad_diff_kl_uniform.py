import torch
import torch.nn.functional as F
from trainer.unlearn.base import UnlearnTrainer
from trainer.unlearn.grad_diff import GradDiff


class GradDiffKLUniform(GradDiff):
    def __init__(self, kl_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        # KL(uniform || model) regularizer over the full vocabulary
        # KL(u || p) = sum_v (1/V) * (log(1/V) - log p(v))
        #            = -log(V) - (1/V) * sum_v log p(v)
        # Minimizing this is equivalent to maximizing the mean log-probability
        # across the vocabulary (i.e., pulling every token's log-prob up).
        logits = forget_outputs.logits                          # (B, T, V)
        labels = forget_inputs["labels"]                 # (B, T)
        vocab_size = logits.size(-1)

        log_probs = F.log_softmax(logits, dim=-1)        # (B, T, V)
        # Mean log-prob across vocab per token; KL(u||p) = -log(V) - mean_log_p
        mean_log_p = log_probs.mean(dim=-1)              # (B, T)
        kl_per_token = -torch.log(torch.tensor(
            float(vocab_size), device=logits.device)) - mean_log_p  # (B, T)

        # Mask out padding / ignored positions (labels == -100 by HF convention)
        valid_mask = (labels != -100).float()            # (B, T)
        kl_loss = (kl_per_token * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

        loss = self.gamma * forget_loss + self.alpha * retain_loss + self.kl_weight * kl_loss

        return (loss, outputs) if return_outputs else loss