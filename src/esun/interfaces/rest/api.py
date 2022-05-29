import os
import sys
import json

from absl import logging
from esun.application.image_service import ImageService
from flask import Blueprint, jsonify, request

bp = Blueprint('api', __name__, url_prefix='')
image_service = ImageService()


@bp.route('/inference', methods=['POST'])
def inference():
    logging.set_verbosity(logging.INFO)
    data = request.get_json(force=True)
    logging.info(json.dumps(data))
    try:
        resp = image_service.get_response(esun_uuid=data['esun_uuid'],
                                          image_64_encoded=data['image'])
    except:
        logging.exception(sys.exc_info()[0])
    output = jsonify(resp)
    logging.info(output.get_data(as_text=True))
    return output
