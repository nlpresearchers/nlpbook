import torch
from torch.nn import CrossEntropyLoss, BCELoss
from torch import nn
import torch.nn.functional as F

class MultipleChoiceQA(nn.Module):
    def __init__(self, pretrain_model):
        super(MultipleChoiceQA, self).__init__()
        self.pretrain_model = pretrain_model
        self.qa_outputs = nn.Linear(pretrain_model.config.hidden_size, 4)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None # size: (batch_size, max_seq_length, 1)
    ):
        outputs = self.pretrain_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)

        return logits