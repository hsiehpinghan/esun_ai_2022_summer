import os
import re
import json
import torch
import numpy as np

from absl import logging
from transformers import AutoTokenizer
from transformers import ElectraForMaskedLM
from scipy.special import softmax


class Util:

    @classmethod
    def get_char_to_similarity_bert_ids(cls):
        with open(file=os.path.join(os.environ['DATA_DIR'], 'char_to_similarity_bert_ids.json'),
                  mode='r') as f:
            char_to_similarity_bert_ids = json.load(fp=f)
        return char_to_similarity_bert_ids
