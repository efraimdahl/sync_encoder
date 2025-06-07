from transformers import PreTrainedModel, PretrainedConfig
import torch
from transformers import PreTrainedModel, BertModel
from transformers import BertModel, BertConfig, PreTrainedModel
import torch
import torch.nn as nn

from transformers import BertModel, BertConfig, PreTrainedModel
import torch
import torch.nn as nn

class RhythmicControlBert(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config, control_dim=48, num_targets=1):
        super().__init__(config)
        self.control_dim = control_dim
        self.num_targets = num_targets

        # Project 48-dim control vector → 768-dim hidden space
        self.input_proj = nn.Linear(control_dim, config.hidden_size)

        # Prebuilt BERT encoder (we use it as a generic transformer)
        self.bert = BertModel(config)

        # Predict a score (or scores) per timestep
        self.output_proj = nn.Linear(config.hidden_size, num_targets)

    def forward(self, control_seq, labels=None):
        """
        control_seq: Tensor [B, 2048, 48]
        labels: Tensor [B, 2048] (or [B, 2048, 1])
        """
        inputs_embeds = self.input_proj(control_seq)  # [B, 2048, 768]
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=control_seq.device)

        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, 2048, 768]

        logits = self.output_proj(hidden_states)  # [B, 2048, num_targets]

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            # labels should match shape [B, 2048, num_targets]
            if labels.ndim == 2:
                labels = labels.unsqueeze(-1)
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

