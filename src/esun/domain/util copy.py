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
    def get_tokenizer(cls):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(os.environ['MODEL_DIR'], '0'))
        return tokenizer

    @classmethod
    def get_add_text_correct_prob_func(cls):
        def add_and_return_char_correct_prob(masked_info, masked_text_probs_mapping, token_id_mapping):
            masked_char = masked_info['masked_char']
            masked_text = masked_info['masked_text']
            probs = masked_text_probs_mapping[masked_text]
            id_0 = token_id_mapping.get(masked_char)
            id_1 = token_id_mapping.get(f'##{masked_char}')
            prob_0 = probs[id_0] if probs[id_0].shape == () else None
            prob_1 = probs[id_1] if probs[id_1].shape == () else None
            if (prob_0 is None) and (prob_1 is None):
                return 0
            if prob_0 is None:
                return prob_1
            if prob_1 is None:
                return prob_0
            char_correct_prob = prob_0 if prob_0 > prob_1 else prob_1
            masked_info['char_correct_prob'] = char_correct_prob
            return char_correct_prob

        add_and_return_char_correct_prob_func = np.vectorize(pyfunc=add_and_return_char_correct_prob,
                                                             signature='(),(),()->()')

        def add_text_correct_prob(text_obj, masked_text_probs_mapping, token_id_mapping):
            char_correct_probs = add_and_return_char_correct_prob_func(masked_info=text_obj['masked_infos'],
                                                                       masked_text_probs_mapping=masked_text_probs_mapping,
                                                                       token_id_mapping=token_id_mapping)
            text_obj['text_correct_prob'] = np.average(a=char_correct_probs)

        add_text_correct_prob_func = np.vectorize(pyfunc=add_text_correct_prob,
                                                  signature='(),(),()->()')

        return add_text_correct_prob_func

    @classmethod
    def get_models(cls):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logging.set_verbosity(logging.INFO)
        logging.info(f'using device({device}).')
        # models = [ElectraForMaskedLM.from_pretrained(pretrained_model_name_or_path=os.path.join(
        #    os.environ['MODEL_DIR'], str(i))).to(device) for i in range(5)]
        models = [ElectraForMaskedLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(os.environ['MODEL_DIR'], '0')).to(device)]
        return models

    @classmethod
    def get_is_char_equal_func(cls):
        def is_char_equal(char_0, char_1):
            return 1 if char_0 == char_1 else 0

        is_char_equal_func = np.vectorize(pyfunc=is_char_equal,
                                          signature='(),()->()')
        return is_char_equal_func

    @classmethod
    def get_token_id_mapping(cls):
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(os.environ['MODEL_DIR'], '0'))
        return tokenizer.get_vocab()

    @classmethod
    def get_char_to_similarity_bert_ids(cls):
        with open(file=os.path.join(os.environ['DATA_DIR'], 'char_to_similarity_bert_ids.json'),
                  mode='r') as f:
            char_to_similarity_bert_ids = json.load(fp=f)
        return char_to_similarity_bert_ids

    @classmethod
    def get_get_probs_func(cls):
        def get_probs(logits, mask_position):
            return softmax(x=logits[mask_position])

        get_probs_func = np.vectorize(pyfunc=get_probs,
                                      signature='(m,n),()->(n)')
        return get_probs_func

    @classmethod
    def get_get_mask_position_func(cls):
        def get_mask_position(input_ids):
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=os.path.join(os.environ['MODEL_DIR'], '0'))
            return np.where(input_ids == tokenizer.mask_token_id)[0][0]

        get_mask_position_func = np.vectorize(pyfunc=get_mask_position,
                                              signature='(n)->()')
        return get_mask_position_func

    @classmethod
    def get_convert_to_text_objs_func(cls):
        pattern_object = re.compile(pattern=r'\s+')
        convert_to_text_objs_func = np.vectorize(pyfunc=lambda text: {'text': pattern_object.sub('', text)},
                                                 signature='()->()')
        return convert_to_text_objs_func

    @classmethod
    def get_add_masked_infos_func(cls):
        def get_masked_info(mask_index, text):
            return {'masked_char': text[mask_index],
                    'masked_text': text[:mask_index]+'[MASK]'+text[mask_index+1:]}

        get_masked_info_func = np.vectorize(pyfunc=get_masked_info,
                                            signature='(),()->()')

        def add_masked_infos(text_obj):
            text = text_obj['text']
            mask_indexes = list(range(len(text)))
            masked_infos = get_masked_info_func(mask_index=mask_indexes,
                                                text=text)
            text_obj['masked_infos'] = masked_infos

        add_masked_infos_func = np.vectorize(pyfunc=add_masked_infos,
                                             signature='()->()')

        return add_masked_infos_func

    @classmethod
    def get_get_masked_texts(cls):
        pattern_object = re.compile(pattern=r'\s+')
        convert_to_text_objs_func = np.vectorize(pyfunc=lambda text: {'text': pattern_object.sub('', text)},
                                                 signature='()->()')
        return convert_to_text_objs_func


if __name__ == '__main__':
    util = Util()
    token_id_mapping = util.get_token_id_mapping()
    print(type(token_id_mapping))
