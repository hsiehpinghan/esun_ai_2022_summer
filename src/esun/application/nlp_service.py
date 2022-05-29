import os
import hashlib

from cache import cache
from typing import List
from datetime import datetime
from esun.domain.nlp_entity import NlpEntity


class NlpService():
    _captain_email = os.environ['CAPTAIN_EMAIL']
    _salt = os.environ['SALT']

    def get_response(self, esun_uuid: str, sentence_list: List[str]):
        answer = cache.get(esun_uuid)
        if answer is None:
            nlp_entity = NlpEntity(id=esun_uuid,
                                   sentence_list=sentence_list)
            answer = nlp_entity.get_answer()
            cache.set(esun_uuid, answer)
        return {'esun_uuid': esun_uuid,
                'server_uuid': self._generate_server_uuid(),
                'server_timestamp': self._generate_server_timestamp(),
                'answer': answer}

    def _generate_server_uuid(self):
        s = hashlib.sha256()
        data = (NlpService._captain_email + str(int(datetime.now().utcnow().timestamp())
                                                ) + NlpService._salt).encode('utf-8')
        s.update(data)
        server_uuid = s.hexdigest()
        return server_uuid

    def _generate_server_timestamp(self):
        return int(datetime.now().utcnow().timestamp())


if __name__ == '__main__':
    from app import app

    cache.init_app(app)
    service = NlpService()
    print(service.get_response(esun_uuid='my_esun_uuid',
                               sentence_list=['轉客服轉接客服接信用卡專員',
                                              '轉克服轉接克服接信用卡專員',
                                              '轉客服轉接客服直接信用卡專員',
                                              '轉客服轉接客服轉接信用卡專員',
                                              '轉克服轉接客服轉接信用卡專員',
                                              '轉克服轉接客服的轉接信用卡專員',
                                              '我轉客服轉接客服轉接信用卡專員',
                                              '轉客服轉接客服的轉接信用卡專員',
                                              '轉客服轉接客服轉接到信用卡專員',
                                              '我轉客服轉接客服的轉接信用卡專員']))
