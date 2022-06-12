import re
import torch
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

    def __init__(self, id: str, sentence_list: List[str], batch_size: int = 16) -> None:
        super().__init__(id=Id(value=id))
        self._sentence_list = sentence_list

    def get_answer(self) -> str:
        chinese_only_sentences = self._get_chinese_only_sentences(
            sentences=self._sentence_list)
        answer = self._get_predict_sentence(
            chinese_only_sentences=chinese_only_sentences)
        return answer

    def _get_chinese_only_sentences(self, sentences):
        chinese_only_sentences = []
        for sentence in sentences:
            chinese_only_sentence = re.sub(pattern='([^\u4e00-\u9fa5])+',
                                           repl='',
                                           string=sentence)
            chinese_only_sentences.append(chinese_only_sentence)
        return chinese_only_sentences

    def _get_predict_sentence(self, chinese_only_sentences):
        texts = self._get_corrected_texts(texts=chinese_only_sentences)
        result_dict = defaultdict(int)
        for text in texts:
            result_dict[text[0]] += 1
        sorted_sentences = sorted(result_dict.items(),
                                  key=lambda x: x[1],
                                  reverse=True)
        return sorted_sentences[0][0]

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
