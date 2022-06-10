import re
import copy
import torch
import numpy as np

from typing import List
from collections import defaultdict
from scipy.special import softmax
from esun.domain.util import Util
from esun.domain.abstract_id import AbstractId
from esun.domain.abstract_entity import AbstractEntity


class Id(AbstractId):

    def __init__(self, value: str) -> None:
        super().__init__(value=value)


class NlpEntity(AbstractEntity):
    _char_to_similarity_bert_ids = Util.get_char_to_similarity_bert_ids()

    def __init__(self, id: str, sentence_list: List[str], batch_size: int = 128) -> None:
        super().__init__(id=Id(value=id))
        self._sentence_list = sentence_list

    def get_answer(self) -> str:
        sentence_objs = [{'sentence': sentence}
                         for sentence in self._sentence_list]
        self._add_chinese_only_sentence(sentence_objs=sentence_objs)
        self._add_no_auxiliary_word_sentence(sentence_objs=sentence_objs)
        pred_sentence = self._get_pred_sentence(sentence_objs=sentence_objs)
        return pred_sentence

    def _get_pred_sentence(self, sentence_objs):
        sentences_dict = defaultdict(list)
        for (i, sentence_obj) in enumerate(sentence_objs):
            no_auxiliary_word_sentence = sentence_obj['no_auxiliary_word_sentence']
            sentences_dict[len(no_auxiliary_word_sentence)].append({'sentence_index': i,
                                                                    'no_auxiliary_word_sentence': no_auxiliary_word_sentence})
        # split list if different too much
        sentence_infos_list = []
        for sentence_length in sentences_dict:
            sentence_infos = self._split_list_if_different_too_much(
                sentences_dict[sentence_length])
            for sentence_info in sentence_infos:
                sentence_infos_list.append(sentence_info)
        masked_sentence_infos = []
        for sentence_infos in sentence_infos_list:
            masked_sentence_info = self._get_masked_sentence_info(
                sentence_infos=sentence_infos)
            masked_sentence_infos.append(masked_sentence_info)
        print(masked_sentence_infos)

    def _get_masked_sentence_info(self, sentence_infos):
        sentences = [sentence_info['no_auxiliary_word_sentence']
                     for sentence_info in sentence_infos]
        masked_sentence = []
        similarity_bert_ids_list = []
        sent = [list(sentence) for sentence in sentences]
        arr = np.array(sent)
        for (i, char_) in enumerate(arr[0]):
            if np.all(arr[:, i] == char_):
                masked_sentence.append(char_)
            else:
                similarity_bert_ids = self._get_similarity_bert_ids(
                    chars=arr[:, i])
                similarity_bert_ids_list.append(similarity_bert_ids)
                masked_sentence.append('[MASK]')
        return {'masked_sentence': ''.join(masked_sentence),
                'similarity_bert_ids_list': similarity_bert_ids_list}

    def _get_similarity_bert_ids(self, chars):
        similarity_bert_ids = set()
        for char_ in chars:
            if char_ not in NlpEntity._char_to_similarity_bert_ids:
                continue
            similarity_bert_ids.update(
                NlpEntity._char_to_similarity_bert_ids[char_])
        return list(similarity_bert_ids)

    def _is_char_equal(self, char_0, char_1):
        return 1 if char_0 == char_1 else 0

    def _split_list_if_different_too_much(self, sentence_infos):
        result = []
        # {'sentence_index': i,
        # 'no_auxiliary_word_sentence': no_auxiliary_word_sentence}
        sentence_infos_tmp = copy.deepcopy(x=sentence_infos)
        while len(sentence_infos_tmp) > 0:
            main_sentence_info = sentence_infos_tmp.pop(-1)
            main_no_auxiliary_word_sentence = main_sentence_info['no_auxiliary_word_sentence']
            result.append([main_sentence_info])
            for i in range(len(sentence_infos_tmp)-1, -1, -1):
                no_auxiliary_word_sentence = sentence_infos_tmp[i]['no_auxiliary_word_sentence']
                assert len(main_no_auxiliary_word_sentence) == len(
                    no_auxiliary_word_sentence)
                scores = []
                for (char_0, char_1) in zip(list(main_no_auxiliary_word_sentence), list(no_auxiliary_word_sentence)):
                    if char_0 == char_1:
                        scores.append(1)
                    else:
                        scores.append(0)
                if np.average(scores) > 0.7:
                    result[-1].append(sentence_infos_tmp.pop(i))
        return result

    def _add_chinese_only_sentence(self, sentence_objs):
        for sentence_obj in sentence_objs:
            sentence_obj['chinese_only_sentence'] = re.sub(pattern='([^\u4e00-\u9fa5])+',
                                                           repl='',
                                                           string=sentence_obj['sentence'])

    def _add_no_auxiliary_word_sentence(self, sentence_objs):
        auxiliary_words = {'哉', '喂', '嘛', '了', '蛤', '呵', '哪', '嗯', '啊', '耶', '欸',
                           '吧', '矣', '呀', '的', '哇', '喔', '歟', '嗎', '啦', '呢', '呃'}
        for sentence_obj in sentence_objs:
            auxiliary_chars = []
            normal_chars = []
            for (i, char_) in enumerate(sentence_obj['chinese_only_sentence']):
                if char_ in auxiliary_words:
                    auxiliary_chars.append((i, char_))
                else:
                    normal_chars.append((i, char_))
            sentence_obj['auxiliary_chars'] = auxiliary_chars
            sentence_obj['no_auxiliary_word_sentence'] = ''.join(
                [char_ for (i, char_) in normal_chars])


if __name__ == '__main__':
    nlp_entity = NlpEntity(id='my_id',
                           sentence_list=['可能 導致 不是 泡沫 再現',
                                          '可能 導致 不是 泡沫 在線',
                                          '可能 導致 不是 泡沫 再 見',
                                          '可能 導致 不是 泡沫 在 現',
                                          '可能 導致 不 是 泡沫 再現',
                                          '可能 導致 不是 泡沫 在 見',
                                          '可能 導致 不是 泡沫 在 限',
                                          '可能 導致 不 是 泡沫 在線',
                                          '可能 導致 不是 泡沫 在 線',
                                          '可能 導致 股市 泡沫 再現'])
    answer = nlp_entity.get_answer()
    print(answer)
