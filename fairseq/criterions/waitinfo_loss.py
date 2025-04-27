# waitinfo_loss.py

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion
import torch.nn.functional as F
import torch
from fairseq.criterions import register_criterion

# Register the criterion with the name 'waitinfo_loss'
@register_criterion('waitinfo_loss')
class WaitInfoLoss(FairseqCriterion):
    def __init__(self, task, label_smoothing=0.0):
        super().__init__(task)
        self.label_smoothing = label_smoothing

    def forward(self, model, sample, reduce=True):
        # Get the model output
        net_output, wait_logits = model(**sample['net_input'])

        # Translation Loss (Cross-entropy loss)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        translation_loss = F.nll_loss(
            lprobs.view(-1, lprobs.size(-1)),
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction='sum' if reduce else 'none',
        )

        # Wait Info Loss (binary cross-entropy)
        wait_targets = sample['wait_targets']  # Shape: (batch, src_len)
        wait_logits = wait_logits.squeeze(-1)  # Shape: (batch, src_len)
        wait_loss = F.binary_cross_entropy_with_logits(wait_logits, wait_targets.float(), reduction='sum')

        # Total Loss: combining both losses
        total_loss = translation_loss + wait_loss

        sample_size = sample['ntokens']
        logging_output = {
            'loss': total_loss.data,
            'translation_loss': translation_loss.data,
            'wait_loss': wait_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }

        return total_loss, sample_size, logging_output
