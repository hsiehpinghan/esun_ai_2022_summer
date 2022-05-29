import csv
import pandas as pd

from esun.domain.image_entity import ImageEntity


class TestImageEntity():

    def test_get_word(self) -> None:
        # self._test_get_word_esun_800_words()
        # self._test_get_word_hadwritting_most_used_words_v2()
        self._test_get_word_error_words()

    def _test_get_word_error_words(self) -> None:
        df = pd.read_csv(
            filepath_or_buffer='/home/hsiehpinghan/git/esun_ai_2021_summer/data/模型訓練資料_e_o_corrected.csv')
        df = df[df['word'] == 'e']
        with open('/tmp/test_get_word_error_words.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for (index, row) in df.iterrows():
                image_64_encoded = row['image_64_encoded']
                image_entity = ImageEntity(id=f'id_{index}',
                                           image_64_encoded=image_64_encoded)
                pred_word = image_entity.get_word()
                writer.writerow([pred_word, image_64_encoded])

    def _test_get_word_esun_800_words(self) -> None:
        df = pd.read_csv(filepath_or_buffer='/home/hsiehpinghan/git/esun_ai_2021_summer/data/模型訓練資料_corrected.csv',
                         index_col='id')
        df = df[(df['word'] != 'e') & (df['word'] != 'o')]
        words = pd.unique(values=df['word'])
        for (index, word) in enumerate(words):
            row = df[df['word'] == word].head(1)
            word = row['word'].values[0]
            if word == 'n':
                word = 'isnull'
            image_64_encoded = row['image_64_encoded'].values[0]
            image_entity = ImageEntity(id=f'id_{index}',
                                       image_64_encoded=image_64_encoded)
            pred_word = image_entity.get_word()
            assert word == pred_word, image_64_encoded

    def _test_get_word_hadwritting_most_used_words_v2(self) -> None:
        df = pd.read_csv(
            filepath_or_buffer='/home/hsiehpinghan/git/esun_ai_2021_summer/data/hadwritting_most_used_words_v2.csv')
        df = df[(df['word'] != 'e') & (df['word'] != 'o')]
        words = pd.unique(values=df['word'])
        labels = set(ImageEntity.LABEL_NAMES)
        for (index, word) in enumerate(words):
            row = df[df['word'] == word].head(1)
            word = row['word'].values[0]
            if word == 'n':
                word = 'isnull'
            if word not in labels:
                word = 'isnull'
            image_64_encoded = row['image_64_encoded'].values[0]
            image_entity = ImageEntity(id=f'id_{index}',
                                       image_64_encoded=image_64_encoded)
            pred_word = image_entity.get_word()
            assert word == pred_word, image_64_encoded
            """
            if word != pred_word:
                print(f'word: {word} / pred_word: {pred_word}')
                print(image_64_encoded)
            """
