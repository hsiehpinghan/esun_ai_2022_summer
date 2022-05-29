import os
import re
import torch
import numpy as np

from typing import List
from transformers import AutoTokenizer
from esun.domain.util import Util
from esun.domain.abstract_id import AbstractId
from esun.domain.abstract_entity import AbstractEntity


class Id(AbstractId):

    def __init__(self, value: str) -> None:
        super().__init__(value=value)


class NlpEntity(AbstractEntity):
    _token_id_mapping = Util.get_token_id_mapping()
    _convert_to_text_objs_func = Util.get_convert_to_text_objs_func()
    _add_masked_infos_func = Util.get_add_masked_infos_func()
    _tokenizer = Util.get_tokenizer()
    _device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _models = Util.get_models()
    _get_mask_position_func = Util.get_get_mask_position_func()
    _get_probs_func = Util.get_get_probs_func()
    _add_text_correct_prob_func = Util.get_add_text_correct_prob_func()

    def __init__(self, id: str, sentence_list: List[str], batch_size: int = 128) -> None:
        super().__init__(id=Id(value=id))
        self._sentence_list = sentence_list
        self._batch_size = batch_size
        self.get_sorted_text_objs_func = np.vectorize(pyfunc=self._get_sorted_text_objs,
                                                      signature='(),(n),()->()')

    def get_answer(self) -> str:
        sorted_text_objs_list = self.get_sorted_text_objs_func(model=self._models,
                                                               texts=self._sentence_list,
                                                               token_id_mapping=NlpEntity._token_id_mapping)
        text_correct_probs_mapping = {}
        for sorted_text_objs in sorted_text_objs_list:
            for sorted_text_obj in sorted_text_objs:
                text = sorted_text_obj['text']
                text_correct_prob = sorted_text_obj['text_correct_prob']
                if text not in text_correct_probs_mapping:
                    text_correct_probs_mapping[text] = []
                text_correct_probs_mapping[text].append(text_correct_prob)
        #answer = sorted_text_objs[0]['text']
        text_average_correct_prob_mapping = {text: np.average(
            a=text_correct_probs_mapping[text]) for text in text_correct_probs_mapping}

        answer = sorted(text_average_correct_prob_mapping.items(),
                        key=lambda x: x[1],
                        reverse=True)[0][0]

        return answer

    def _get_sorted_text_objs(self, model, texts, token_id_mapping):
        print(len(texts))
        text_objs = self._convert_to_text_objs_func(text=texts)
        self._add_masked_infos_func(text_obj=text_objs)
        masked_texts = self._get_masked_texts(text_objs=text_objs)
        inputs = self._get_inputs(masked_texts=masked_texts)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        assert len(input_ids) == len(attention_mask)
        sub_probs_lists = []
        for i in range(0, len(input_ids), self._batch_size):
            sub_input_ids = input_ids[i:i+self._batch_size].to(self._device)
            sub_attention_mask = attention_mask[i:i +
                                                self._batch_size].to(self._device)
            with torch.no_grad():
                sub_outputs = model(input_ids=sub_input_ids,
                                    attention_mask=sub_attention_mask,
                                    labels=sub_input_ids)
            (_, sub_logits_list) = sub_outputs[:]
            sub_input_ids = sub_input_ids.cpu()
            sub_attention_mask = sub_attention_mask.cpu()
            sub_logits_list = sub_logits_list.cpu().numpy()
            sub_mask_positions = self._get_mask_position_func(
                input_ids=sub_input_ids)
            sub_probs_list = self._get_probs_func(logits=sub_logits_list,
                                                  mask_position=sub_mask_positions)
            sub_probs_lists.append(sub_probs_list)
        probs_list = np.concatenate(sub_probs_lists,
                                    axis=0)
        masked_text_probs_mapping = {masked_text: probs for (
            masked_text, probs) in zip(masked_texts, probs_list)}
        self._add_text_correct_prob_func(text_obj=text_objs,
                                         masked_text_probs_mapping=masked_text_probs_mapping,
                                         token_id_mapping=token_id_mapping)
        sorted_text_objs = sorted(text_objs,
                                  key=lambda text_obj: text_obj['text_correct_prob'],
                                  reverse=True)
        return sorted_text_objs

    def _get_masked_texts(self, text_objs):
        masked_texts = list(set([masked_info['masked_text']
                            for text_obj in text_objs for masked_info in text_obj['masked_infos']]))
        return masked_texts

    def _get_inputs(self, masked_texts):
        batch_encoding = self._tokenizer.batch_encode_plus(batch_text_or_text_pairs=masked_texts,
                                                           add_special_tokens=True,
                                                           padding=True,
                                                           return_tensors='pt')
        input_ids = batch_encoding['input_ids']
        assert len(input_ids) == len(masked_texts)
        attention_mask = batch_encoding['attention_mask']
        assert len(attention_mask) == len(masked_texts)
        return {'input_ids': input_ids,
                'attention_mask': attention_mask}


if __name__ == '__main__':
    nlp_entity = NlpEntity(id='my_id',
                           sentence_list=['轉客服轉接客服接信用卡專員',
                                          '轉克服轉接客服接信用卡專員',
                                          '轉客服轉接客服直接信用卡專員',
                                          '轉客服轉接客服轉接信用卡專員',
                                          '轉克服轉接客服轉接信用卡專員',
                                          '轉克服轉接客服的轉接信用卡專員',
                                          '我轉客服轉接客服轉接信用卡專員',
                                          '轉客服轉接客服的轉接信用卡專員',
                                          '轉客服轉接客服轉接到信用卡專員',
                                          '我轉客服轉接客服的轉接信用卡專員'])
    answer = nlp_entity.get_answer()
    print(answer)
