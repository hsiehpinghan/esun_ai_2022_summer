FROM python:3.7.10-stretch
ENV FLASK_APP=/app.py
ENV FLASK_ENV=production
ENV CAPTAIN_EMAIL=thank.hsiehpinghan@gmail.com
ENV SALT=671224
ENV MODEL_DIR=/model
ENV DATA_DIR=/data
ADD ./data /data
ADD ./src/esun /esun
ADD ./src/app.py /app.py
ADD ./src/cache.py /cache.py
ADD ./setup.py /
RUN pip install -e /
ADD ./model /model
CMD flask run --host=0.0.0.0 >> /log/esun_ai_2022_summer.log 2>&1