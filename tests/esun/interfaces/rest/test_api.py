import csv
import json
import time
import aiohttp
import asyncio
import pandas as pd


class TestApi():
    URL = 'http://35.194.149.159/inference'
    ESUN_UUID_PREFIX = '6465e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf'
    ESUN_TIMESTAMP = 1590493849
    RETRY = 2

    def test_compression(self) -> None:
        df = pd.read_csv(
            filepath_or_buffer='/home/hsiehpinghan/git/esun_ai_2021_summer/data/模型訓練資料_corrected.csv')
        df = df[(df['word'] != 'e') & (df['word'] != 'o')]
        loop = asyncio.get_event_loop()
        tasks = []
        for (index, row) in df.iterrows():
            id = row.name
            word = row['word']
            image = row['image_64_encoded']
            task = asyncio.ensure_future(self._measure_time(id=id,
                                                            word=word,
                                                            image=image))
            tasks.append(task)
            if (index + 1) % 5 == 0:
                loop.run_until_complete(asyncio.wait(tasks))
                time.sleep(1)

    async def _write_result(self, id, word, pred_word, start_time, end_time):
        with open('/tmp/compression_test_esun_800.csv', 'a') as csv_file:
            writer = csv.writer(csv_file)
            duration = end_time - start_time
            writer.writerow(
                [id, word, pred_word, start_time, end_time, duration])

    async def _request(self, url, esun_uuid, esun_timestamp, image, retry):
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url,
                                    json={'url': url,
                                          'esun_uuid': esun_uuid,
                                          'esun_timestamp': esun_timestamp,
                                          'image': image,
                                          'retry': retry}) as resp:
                return await resp.json()

    async def _measure_time(self, id, word, image):
        start_time = time.time()
        resp = await self._request(TestApi.URL, f'{TestApi.ESUN_UUID_PREFIX}_{id}', TestApi.ESUN_TIMESTAMP, image, TestApi.RETRY)
        await self._write_result(id=id,
                                 word=word,
                                 pred_word=resp['answer'],
                                 start_time=start_time,
                                 end_time=time.time())
