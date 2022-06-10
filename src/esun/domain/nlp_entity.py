import re
import torch

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

    def __init__(self, id: str, sentence_list: List[str], batch_size: int = 128) -> None:
        super().__init__(id=Id(value=id))
        self._sentence_list = sentence_list

    def get_answer(self) -> str:
        chinese_only_sentences = self._get_chinese_only_sentences(
            sentences=self._sentence_list)
        results = self._get_results(
            chinese_only_sentences=chinese_only_sentences)
        result_dict = defaultdict(int)
        for result in results:
            result_dict[result] += 1
        sorted_sentences = sorted(result_dict.items(),
                                  key=lambda x: x[1],
                                  reverse=True)
        return sorted_sentences[0][0]

    def _get_results(self, chinese_only_sentences):
        with torch.no_grad():
            outputs = self._model(**self._tokenizer(chinese_only_sentences,
                                                    padding=True,
                                                    return_tensors='pt').to(self._device))
        results = []
        for (ids, text) in zip(outputs.logits, chinese_only_sentences):
            _text = self._tokenizer.decode(torch.argmax(
                ids, dim=-1), skip_special_tokens=True).replace(' ', '')
            result = _text[:len(text)]
            results.append(result)
        return results

    def _get_chinese_only_sentences(self, sentences):
        chinese_only_sentences = []
        for sentence in sentences:
            chinese_only_sentence = re.sub(pattern='([^\u4e00-\u9fa5])+',
                                           repl='',
                                           string=sentence)
            chinese_only_sentences.append(chinese_only_sentence)
        return chinese_only_sentences


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
