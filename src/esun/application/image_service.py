import os
import hashlib

from cache import cache
from datetime import datetime
from esun.domain.image_entity import ImageEntity


class ImageService():
    CAPTAIN_EMAIL = os.environ['CAPTAIN_EMAIL']
    SALT = os.environ['SALT']

    def get_response(self, esun_uuid: str, image_64_encoded: str):
        word = cache.get(esun_uuid)
        if word is None:
            image_entity = ImageEntity(id=esun_uuid,
                                       image_64_encoded=image_64_encoded)
            word = image_entity.get_word()
            cache.set(esun_uuid, word)
        return {'esun_uuid': esun_uuid,
                'server_uuid': self._generate_server_uuid(),
                'server_timestamp': self._generate_server_timestamp(),
                'answer': word}

    def _generate_server_uuid(self):
        s = hashlib.sha256()
        data = (ImageService.CAPTAIN_EMAIL + str(int(datetime.now().utcnow().timestamp())
                                                 ) + ImageService.SALT).encode("utf-8")
        s.update(data)
        server_uuid = s.hexdigest()
        return server_uuid

    def _generate_server_timestamp(self):
        return int(datetime.now().utcnow().timestamp())


if __name__ == '__main__':
    service = ImageService()
    print(service.get_response(esun_uuid='my_esun_uuid',
                               image_64_encoded='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'))
