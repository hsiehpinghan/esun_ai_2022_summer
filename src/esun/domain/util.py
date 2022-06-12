import os
import re
import json
import torch
import numpy as np

from absl import logging
from transformers import AutoTokenizer
from transformers import BertForMaskedLM
from esun.domain.ensemble_model import EnsembleModel


class Util:

    @classmethod
    def get_tokenizer(cls):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(os.environ['MODEL_DIR'], 'sub_model_0'))
        return tokenizer

    @classmethod
    def get_model(cls):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logging.set_verbosity(logging.INFO)
        logging.info(f'using device({device}).')
        sub_models = [os.path.join(
            os.environ['MODEL_DIR'], f'sub_model_{i}') for i in range(5)]
        model = EnsembleModel(pretrained_model_name_or_paths=sub_models,
                              device=device)
        return model
