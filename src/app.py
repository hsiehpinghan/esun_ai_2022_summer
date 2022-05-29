from flask import Flask
from cache import cache
from esun.interfaces.rest import (
    api
)

config = {
    'CACHE_TYPE': 'RedisCache',
    'CACHE_DEFAULT_TIMEOUT': 6*60*60,
    'CACHE_REDIS_HOST': 'redis',
    'CACHE_REDIS_PORT': 6379
}

app = Flask(__name__)
app.register_blueprint(api.bp)
app.config.from_mapping(config)
cache.init_app(app)
