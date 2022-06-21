import re
import torch
import numpy as np
import operator

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
    _device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _tokenizer = Util.get_tokenizer()
    _model = Util.get_model()
    _char_to_similarity_bert_ids = Util.get_char_to_similarity_bert_ids()

    def __init__(self, id: str, sentence_list: List[str]) -> None:
        super().__init__(id=Id(value=id))
        self._sentence_list = sentence_list

    def get_answer(self) -> str:
        chinese_only_sentences = self._get_chinese_only_sentences(
            sentences=self._sentence_list)
        sentences_list_similar = self._get_sentences_list_similar(
            sentence_list=chinese_only_sentences)
        sentences_similar = self._get_sentences_similar(
            sentences_list_similar=sentences_list_similar)
        similarity_bert_ids_list = self._get_similarity_bert_ids_list(
            sentences_similar=sentences_similar)
        answer = self._get_predict_sentence(sentences_similar=sentences_similar,
                                            similarity_bert_ids_list=similarity_bert_ids_list)
        return answer

    def _get_most_possible_sentences(self, sentences_similar, similarity_bert_ids_list):
        with torch.no_grad():
            probs_list = self._model(**self._tokenizer(sentences_similar,
                                                       padding=True,
                                                       return_tensors='pt').to(self._device))
        result = []
        assert len(probs_list) == len(
            sentences_similar), f'{len(probs_list)} / {len(sentences_similar)}'
        assert len(probs_list) == len(
            similarity_bert_ids_list), f'{len(probs_list)} / {len(similarity_bert_ids_list)}'
        for (probs, sentence_similar, similarity_bert_ids) in zip(probs_list, sentences_similar, similarity_bert_ids_list):
            assert len(sentence_similar) <= (
                len(probs)-2), f'{len(sentence_similar)} / {len(probs)-2}'
            chars = []
            ps = []
            for (i, char_) in enumerate(sentence_similar):
                if i not in similarity_bert_ids:
                    chars.append(char_)
                else:
                    similarity_bert_id = similarity_bert_ids[i][torch.argmax(
                        probs[[i+1], similarity_bert_ids[i]])]

                    # if char_ == '銀':
                    #    print(char_, similarity_bert_id, similarity_bert_ids[i])

                    char_ = self._tokenizer.decode(similarity_bert_id)
                    ps.append(probs[[i+1], similarity_bert_id].cpu().numpy())
                    chars.append(char_)
            result.append((''.join(chars), np.average(ps)))
        return sorted(result,
                      key=lambda x: x[1],
                      reverse=True)

    def _get_predict_sentence(self, sentences_similar, similarity_bert_ids_list):
        most_possible_sentences = self._get_most_possible_sentences(sentences_similar=sentences_similar,
                                                                    similarity_bert_ids_list=similarity_bert_ids_list)
        return most_possible_sentences[0][0]

    def _get_similarity_bert_ids_list(self, sentences_similar):
        similarity_bert_ids_list = [{} for _ in sentences_similar]
        for (sentence_index, sentence_similar) in enumerate(sentences_similar):
            for (char_index, diff_char) in enumerate(sentence_similar):
                if diff_char not in self._char_to_similarity_bert_ids:
                    similarity_bert_ids = [100]
                else:
                    similarity_bert_ids = self._char_to_similarity_bert_ids[diff_char]
                    if len(similarity_bert_ids) <= 0:
                        similarity_bert_ids = [100]
                similarity_bert_ids_list[sentence_index][char_index] = similarity_bert_ids
                """
                if diff_char == '現':
                    print(diff_char, similarity_bert_ids)
                    #print(i, diff_char_index)
                    #print(similarity_bert_ids_list[i][diff_char_index])
                """
        return similarity_bert_ids_list

    def _get_sentences_similar(self, sentences_list_similar):
        return [sentence_similar for sentences_similar in sentences_list_similar for sentence_similar in sentences_similar]

    def _get_sentences_list_similar(self, sentence_list):
        sentences_dict = defaultdict(list)
        for sentence in sentence_list:
            sentences_dict[len(sentence)].append(sentence)
        sentences_list = sentences_dict.values()
        # filter len >= 1
        sentences_list = [
            sentences for sentences in sentences_list if len(sentences) >= 1]
        # split list if different too much
        sentences_list_similar = []
        for sentences in sentences_list:
            sentences_list_similar += self._split_list_if_different_too_much(
                sentences=sentences)
        # filter len >= 1
        sentences_list_similar = [
            sentences for sentences in sentences_list_similar if len(sentences) >= 1]
        return sentences_list_similar

    def _split_list_if_different_too_much(self, sentences):
        result = []
        sentences_tmp = sentences.copy()
        while len(sentences_tmp) > 0:
            main_ele = sentences_tmp.pop(-1)
            result.append([main_ele])
            is_char_equals = []
            for i in range(len(sentences_tmp)-1, -1, -1):
                assert len(main_ele) == len(sentences_tmp[i])
                for (char_0, char_1) in zip(list(main_ele), list(sentences_tmp[i])):
                    is_char_equals.append(self.is_char_equal(char_0=char_0,
                                                             char_1=char_1))
                if np.average(is_char_equals) > 0.5:
                    result[-1].append(sentences_tmp.pop(i))
        return result

    def is_char_equal(self, char_0, char_1):
        return 1 if char_0 == char_1 else 0

    def _get_chinese_only_sentences(self, sentences):
        chinese_only_sentences = []
        for sentence in sentences:
            chinese_only_sentence = re.sub(pattern='([^\u4e00-\u9fa5])+',
                                           repl='',
                                           string=sentence)
            chinese_only_sentences.append(chinese_only_sentence)
        return chinese_only_sentences

    def _get_corrected_texts(self, texts):
        with torch.no_grad():
            probs = self._model(**self._tokenizer(texts,
                                                  padding=True,
                                                  return_tensors='pt').to(self._device))

        def get_errors(corrected_text, origin_text):
            sub_details = []
            for i, ori_char in enumerate(origin_text):
                """
                if ori_char in [' ', '“', '”', '‘', '’', '琊', '\n', '…', '—', '擤']:
                    # add unk word
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
                    continue
                """
                if i >= len(corrected_text):
                    continue
                if ori_char != corrected_text[i]:
                    if ori_char.lower() == corrected_text[i]:
                        # pass english upper char
                        corrected_text = corrected_text[:i] + \
                            ori_char + corrected_text[i + 1:]
                        continue
                    sub_details.append((ori_char, corrected_text[i], i, i + 1))
            sub_details = sorted(sub_details, key=operator.itemgetter(2))
            return corrected_text, sub_details

        result = []
        for (prob, text) in zip(probs, texts):
            _text = self._tokenizer.decode(torch.argmax(
                prob, dim=-1), skip_special_tokens=True).replace(' ', '')
            corrected_text = _text[:len(text)]
            corrected_text, details = get_errors(corrected_text, text)
            result.append((corrected_text, details))
        return result


if __name__ == '__main__':
    nlp_entity = NlpEntity(id='my_id',
                           sentence_list=['並提升內部監督機制',
                                          '並提升那不見都機制',
                                          '並提升內部間都機制',
                                          '並提升內部件都機制',
                                          '並提升那不間都機制',
                                          '並提升那不監督機制',
                                          '並提升那部監督機制',
                                          '並提升那布建都機制',
                                          '並提升內不見都機制',
                                          '並提升那不件都機制'])
    answer = nlp_entity.get_answer()
    print(answer)
