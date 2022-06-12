import torch
import torch.nn as nn

from typing import Tuple
from typing import Union
from typing import Optional
from transformers import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput


class EnsembleModel(nn.Module):
    def __init__(self, pretrained_model_name_or_paths, device):
        super().__init__()
        models = []
        for pretrained_model_name_or_path in pretrained_model_name_or_paths:
            model = BertForMaskedLM.from_pretrained(
                pretrained_model_name_or_path=pretrained_model_name_or_path).to(device)
            model.eval()
            models.append(model)
        self._models = models

    def forward(self,
                input_ids: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                token_type_ids: Optional[torch.Tensor],
                correction_labels: Optional[torch.Tensor] = None,
                detection_labels: Optional[torch.Tensor] = None,
                output_hidden_states: Optional[bool] = True,
                return_dict: Optional[bool] = True,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        sub_probs_list = []
        for model in self._models:
            outputs = model.forward(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    encoder_hidden_states=encoder_hidden_states,
                                    encoder_attention_mask=encoder_attention_mask,
                                    labels=correction_labels,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict)
            sub_probs = torch.softmax(outputs.logits,
                                      dim=-1)
            sub_probs_list.append(sub_probs)
        probs = torch.stack(sub_probs_list,
                            dim=0)\
            .mean(dim=0)
        return probs
