import os
import re
import json
import torch
import numpy as np

from absl import logging
from transformers import BertTokenizer
from transformers import BertForMaskedLM


class Util:

    @classmethod
    def get_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.environ['MODEL_DIR'])
        return tokenizer

    @classmethod
    def get_model(cls):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.set_verbosity(logging.INFO)
        logging.info(f'using device({device}).')
        model = BertForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=os.environ['MODEL_DIR']).to(device)
        return model
