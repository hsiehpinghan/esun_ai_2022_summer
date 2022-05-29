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
    _token_id_mapping = Util.get_token_id_mapping()
    _convert_to_text_objs_func = Util.get_convert_to_text_objs_func()
    _add_masked_infos_func = Util.get_add_masked_infos_func()
    _tokenizer = Util.get_tokenizer()
    _device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _models = Util.get_models()
    _get_mask_position_func = Util.get_get_mask_position_func()
    _get_probs_func = Util.get_get_probs_func()
    _add_text_correct_prob_func = Util.get_add_text_correct_prob_func()
    _is_char_equal_func = Util.get_is_char_equal_func()
    _char_to_similarity_bert_ids = Util.get_char_to_similarity_bert_ids()

    def __init__(self, id: str, sentence_list: List[str], batch_size: int = 128) -> None:
        super().__init__(id=Id(value=id))
        self._sentence_list = sentence_list
        self._batch_size = batch_size
        self._get_sorted_text_objs_func = np.vectorize(pyfunc=self._get_sorted_text_objs,
                                                       signature='(),(n),()->()')
        self._get_most_likely_sentences_func = np.vectorize(pyfunc=self._get_most_likely_sentences,
                                                            signature='(),(n)->(n)')

    def get_answer(self) -> str:
        sentence_list = [sentence.replace(' ', '')
                         for sentence in self._sentence_list]
        similar_text_objs = self._get_similar_text_objs(
            sentence_list=sentence_list)
        if len(similar_text_objs) > 0:
            most_likely_sentences_list = self._get_most_likely_sentences_func(model=self._models,
                                                                              similar_text_objs=similar_text_objs)
        sentence_list = list(
            set(sentence_list+list(most_likely_sentences_list.flatten())))
        sorted_text_objs_list = self._get_sorted_text_objs_func(model=self._models,
                                                                texts=sentence_list,
                                                                token_id_mapping=NlpEntity._token_id_mapping)
        text_correct_probs_mapping = {}
        for sorted_text_objs in sorted_text_objs_list:
            for sorted_text_obj in sorted_text_objs:
                text = sorted_text_obj['text']
                text_correct_prob = sorted_text_obj['text_correct_prob']
                if text not in text_correct_probs_mapping:
                    text_correct_probs_mapping[text] = []
                text_correct_probs_mapping[text].append(text_correct_prob)
        text_average_correct_prob_mapping = {text: np.average(
            a=text_correct_probs_mapping[text]) for text in text_correct_probs_mapping}

        answer = sorted(text_average_correct_prob_mapping.items(),
                        key=lambda x: x[1],
                        reverse=True)[0][0]

        return answer

    def get_answer_v2(self) -> str:
        sentence_list = [sentence.replace(' ', '')
                         for sentence in self._sentence_list]
        similar_text_objs = self._get_similar_text_objs(
            sentence_list=sentence_list)
        if len(similar_text_objs) > 0:
            most_likely_sentences_list = self._get_most_likely_sentences_func(model=self._models,
                                                                              similar_text_objs=similar_text_objs)
        return most_likely_sentences_list

    def _get_most_likely_sentences(self, model, similar_text_objs):
        most_likely_sentences = []
        masked_sentences = [similar_text_obj['masked_sentence']
                            for similar_text_obj in similar_text_objs]
        inputs = self._get_inputs(masked_texts=masked_sentences)
        input_ids = inputs['input_ids'].to(self._device)
        attention_mask = inputs['attention_mask'].to(self._device)
        assert len(input_ids) == len(attention_mask)
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids)
        (_, logits_list) = outputs[:]
        input_ids = input_ids.cpu()
        attention_mask = attention_mask.cpu()
        logits_list = logits_list.cpu().numpy()  # shape: (3, 17, 21128)
        bert_ids_list_list = [similar_text_obj['bert_ids_list']
                              for similar_text_obj in similar_text_objs]
        assert len(logits_list) == len(bert_ids_list_list)
        probs_list_list = softmax(x=logits_list)
        most_likely_sentences = []
        for (probs_list, bert_ids_list) in zip(probs_list_list, bert_ids_list_list):
            # len(probs_list) / len(bert_ids_list): 17 / 13
            best_bert_ids = []
            for i in range(len(bert_ids_list)):
                bert_ids = bert_ids_list[i]
                bert_ids_probs = np.take(a=probs_list[i+1],
                                         indices=bert_ids)
                best_bert_id_index = np.argmax(a=bert_ids_probs)
                best_bert_id = bert_ids[best_bert_id_index]
                best_bert_ids.append(best_bert_id)
            most_likely_sentences.append(self._tokenizer.decode(token_ids=best_bert_ids,
                                                                skip_special_tokens=True).replace(' ', ''))
        return np.array(most_likely_sentences)

        """
        mask_positions_list = []
        for inp_ids in input_ids:
            mask_positions = self._get_mask_positions(input_ids=inp_ids)
            mask_positions_list.append(mask_positions)
        assert len(input_ids) == len(logits_list) == len(
            mask_positions_list) == len(similar_text_objs)
        for (inp_ids, logits, mask_positions, similar_text_obj) in zip(input_ids, logits_list, mask_positions_list, similar_text_objs):
            probs_list = self._get_probs_list(logits, mask_positions)
            assert len(mask_positions) == len(probs_list) == len(
                similar_text_obj['similarity_bert_ids_list'])
            for (mask_position, probs, similarity_bert_ids) in zip(mask_positions, probs_list, similar_text_obj['similarity_bert_ids_list']):
                if len(similarity_bert_ids) > 0:
                    similarity_bert_ids_probs = np.take(a=probs,
                                                        indices=similarity_bert_ids)
                    most_similarity_index = np.argmax(
                        a=similarity_bert_ids_probs)
                    most_similarity_bert_id = similarity_bert_ids[most_similarity_index]
                else:
                    most_similarity_bert_id = self._tokenizer.unk_token_id
                inp_ids[mask_position] = most_similarity_bert_id
            most_likely_sentences.append(self._tokenizer.decode(token_ids=inp_ids,
                                                                skip_special_tokens=True).replace(' ', ''))
        return np.array(most_likely_sentences)
        """

    def _get_probs_list(self, logits, mask_positions):
        probs_list = []
        for mask_position in mask_positions:
            probs = softmax(x=logits[mask_position])
            probs_list.append(probs)
        return probs_list

    def _get_mask_positions(self, input_ids):
        return np.where(input_ids == self._tokenizer.mask_token_id)[0]

    def _get_similar_text_objs(self, sentence_list):
        sentences_dict = defaultdict(list)
        for sentence in sentence_list:
            sentences_dict[len(sentence)].append(sentence)
        sentences_list = sentences_dict.values()
        # filter len < 0
        sentences_list = [
            sentences for sentences in sentences_list if len(sentences) > 1]
        # split list if different too much
        sentences_list_similar = []
        for sentences in sentences_list:
            sentences_list_similar += self._split_list_if_different_too_much(
                sentences=sentences)
        # filter len < 0
        similar_text_objs = []
        sentences_list_similar = [
            sentences for sentences in sentences_list_similar if len(sentences) > 1]
        for sentences_similar in sentences_list_similar:
            similar_text_obj = self._get_similar_text_obj(
                sentences=sentences_similar)
            similar_text_objs.append(similar_text_obj)
        return similar_text_objs

    def _get_similar_text_obj(self, sentences):
        masked_sentence = []
        bert_ids_list = []
        sent = [list(sentence) for sentence in sentences]
        arr = np.array(sent)
        for (i, char_) in enumerate(arr[0]):
            if np.all(arr[:, i] == char_):
                masked_sentence.append(char_)
                bert_ids = self._get_bert_ids(char_)
                bert_ids_list.append(bert_ids)
            else:
                masked_sentence.append('[MASK]')
                similarity_bert_ids = self._get_similarity_bert_ids(
                    chars=arr[:, i])
                bert_ids_list.append(similarity_bert_ids)
        return {'sentences': sentences,
                'masked_sentence': ''.join(masked_sentence),
                'bert_ids_list': bert_ids_list}

    def _get_similarity_bert_ids(self, chars):
        similarity_bert_ids = set()
        for char_ in chars:
            if char_ not in self._char_to_similarity_bert_ids:
                continue
            similarity_bert_ids.update(
                self._char_to_similarity_bert_ids[char_])
        return list(similarity_bert_ids)

    def _get_bert_ids(self, char):
        bert_ids = []
        if char in self._token_id_mapping:
            bert_ids.append(self._token_id_mapping.get(char))
        char_tmp = f'##{char}'
        if char_tmp in self._token_id_mapping:
            bert_ids.append(self._token_id_mapping.get(char_tmp))
        if len(bert_ids) <= 0:
            bert_ids.append(self._tokenizer.unk_token_id)
        return bert_ids

    def _split_list_if_different_too_much(self, sentences):
        result = []
        sentences_tmp = sentences.copy()
        while len(sentences_tmp) > 0:
            main_ele = sentences_tmp.pop(-1)
            result.append([main_ele])
            for i in range(len(sentences_tmp)-1, -1, -1):
                is_char_equals = self._is_char_equal_func(char_0=list(main_ele),
                                                          char_1=list(sentences_tmp[i]))
                if np.average(is_char_equals) > 0.8:
                    result[-1].append(sentences_tmp.pop(i))
        return result

    def _get_sorted_text_objs(self, model, texts, token_id_mapping):
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
                                          '轉克服轉接克服接信用卡專員',
                                          '轉客服轉接客服直接信用卡專員',
                                          '轉客服轉接客服轉接信用卡專員',
                                          '轉克服轉接客服轉接信用卡專員',
                                          '轉克服轉接客服的轉接信用卡專員',
                                          '我轉客服轉接客服轉接信用卡專員',
                                          '轉客服轉接客服的轉接信用卡專員',
                                          '轉客服轉接客服轉接到信用卡專員',
                                          '我轉客服轉接客服的轉接信用卡專員'])
    answer = nlp_entity.get_answer_v2()
    print(answer)
