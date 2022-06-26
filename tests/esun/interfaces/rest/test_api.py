import csv
import json
import time
import aiohttp
import asyncio

from tqdm import tqdm


class TestApi():
    URL = 'http://35.194.149.173:20180/inference'
    ESUN_UUID_PREFIX = '6465e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf'
    ESUN_TIMESTAMP = 1590493849
    RETRY = 2

    def test_compression(self) -> None:
        with open(file='/home/hsiehpinghan/Downloads/2022summer_train_data/train_all.json',
                  mode='r') as f:
            items = json.load(fp=f)

        loop = asyncio.get_event_loop()
        tasks = []
        for (i, item) in tqdm(enumerate(items),
                              total=len(items)):
            task = asyncio.ensure_future(self._measure_time(item=item))
            tasks.append(task)
            if (i + 1) % 10 == 0:
                loop.run_until_complete(asyncio.wait(tasks))
                time.sleep(1)

    async def _write_result(self, id, ground_truth_sentence, answer, start_time, end_time):
        with open('/tmp/compression_test.csv', 'a') as csv_file:
            writer = csv.writer(csv_file)
            duration = end_time - start_time
            writer.writerow(
                [id, ground_truth_sentence, answer, start_time, end_time, duration])

    async def _request(self, url, esun_uuid, esun_timestamp, sentence_list, phoneme_sequence_list, retry):
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url,
                                    json={'esun_uuid': esun_uuid,
                                          'esun_timestamp': esun_timestamp,
                                          'sentence_list': sentence_list,
                                          'phoneme_sequence_list': phoneme_sequence_list,
                                          'retry': retry}) as resp:
                return await resp.json()

    async def _measure_time(self, item):
        ground_truth_sentence = item['ground_truth_sentence']
        id = item['id']
        sentence_list = item['sentence_list']
        phoneme_sequence_list = item['phoneme_sequence_list']
        start_time = time.time()
        resp = await self._request(url=TestApi.URL,
                                   esun_uuid=f'{TestApi.ESUN_UUID_PREFIX}_{id}',
                                   esun_timestamp=TestApi.ESUN_TIMESTAMP,
                                   sentence_list=sentence_list,
                                   phoneme_sequence_list=phoneme_sequence_list,
                                   retry=TestApi.RETRY)
        await self._write_result(id=id,
                                 ground_truth_sentence=ground_truth_sentence,
                                 answer=resp['answer'],
                                 start_time=start_time,
                                 end_time=time.time())
