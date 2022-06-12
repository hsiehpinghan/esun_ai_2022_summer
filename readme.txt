# reference
https://tbrain.trendmicro.com.tw/Competitions/Details/23

# setup redis
vi /etc/hosts
  127.0.0.1       redis

# create miniconda virtual machine (linux)
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
/home/hsiehpinghan/miniconda3/bin/conda create --name esun_ai_2022_summer python=3.7.10
export MINICONDA_HOME=/home/hsiehpinghan/miniconda3
export PATH=$MINICONDA_HOME/bin:$PATH
conda init bash
cd /home/hsiehpinghan/git/esun_ai_2022_summer/notebook
conda activate esun_ai_2022_summer
pip install --upgrade pip
pip install absl-py==1.0.0
pip install scipy==1.4.1
pip install gunicorn==20.1.0
pip install Flask==2.0.3
pip install Flask-Caching==1.11.1
pip install redis==3.5.3
pip install torch==1.11.0
pip install transformers==4.18.0

# start miniconda jupyter notebook (linux)
export ANACONDA_HOME=/home/hsiehpinghan/anaconda3
export PATH=$ANACONDA_HOME/bin:$PATH
cd /home/hsiehpinghan/git/esun_ai_2022_summer/notebook
conda activate esun_ai_2022_summer
jupyter notebook

# install package
cd /home/hsiehpinghan/git/esun_ai_2022_summer
pip install -e .

# test rest
export FLASK_APP=/home/hsiehpinghan/git/esun_ai_2022_summer/src/app.py
export FLASK_ENV=production
export CAPTAIN_EMAIL=thank.hsiehpinghan@gmail.com
export SALT=671224
export MODEL_DIR=/home/hsiehpinghan/git/esun_ai_2022_summer/model
export DATA_DIR=/home/hsiehpinghan/git/esun_ai_2022_summer/data
cd /home/hsiehpinghan/git/esun_ai_2022_summer/src
#gunicorn --workers=1 --bind=0.0.0.0:5000 app:app > /tmp/esun_ai_2022_summer.log 2>&1
flask run --host=0.0.0.0 >> /tmp/esun_ai_2022_summer.log 2>&1
curl -v -X POST -H "Content-Type: application/json" -d @/home/hsiehpinghan/git/esun_ai_2022_summer/data/request.json http://localhost:5000/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}

# kill app
jobs -l
kill ???

# uninstall app
pip uninstall esun-ai-2022-summer

# build docker image
cd /home/hsiehpinghan/git/esun_ai_2022_summer
docker image build -t hsiehpinghan/esun_ai_2022_summer:5.0.0 .

# push to docker hub
docker login
docker push hsiehpinghan/esun_ai_2022_summer:5.0.0

# run gcp container
## Machine type: n1-highcpu-8 (8 vCPU, 8 GB memory)
## Boot disk: 50 GB
sudo apt-get install \
     ca-certificates \
     curl \
     gnupg \
     lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker thank_hsiehpinghan
newgrp docker
mkdir -p /tmp/redis/data
docker run --name redis \
  --restart=always \
  -p 6379:6379 \
  -v /tmp/redis/data:/data \
  -itd \
  redis:6.2.4 redis-server --appendonly yes
sudo mkdir -p /log/log_0
docker login
docker run --name esun_ai_2022_summer_0 \
  --restart=always \
  -e "TZ=Asia/Taipei" \
  -p 10180:5000 \
  -v /log/log_0:/log \
  --link redis:redis \
  -td hsiehpinghan/esun_ai_2022_summer:5.0.0
curl -v -X POST -H "Content-Type: application/json" -d '{"esun_uuid": "add61efb7e8d9268b972b95b6fa53db93780b6b22fbf","esun_timestamp": 1590493849,"sentence_list": ["喂 你好 密碼 我 要 進去","喂 你好 密碼 哇 進去","喂 你好 密碼 的 話 進去","喂 您好 密碼 我 要 進去","喂 你好 密碼 無法 進去","喂 你好 密碼 waa 進去","喂 你好 密碼 while 進去","喂 你好 密碼 文化 進去","喂 你好 密碼 挖 進去","喂 您好 密碼 哇 進去"],"phoneme_sequence_list": ["w eI4 n i:3 x aU4 m i:4 m A:3 w O:3 j aU1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 w A:1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 t ax5 x w A:4 ts6 j ax n4 ts6_h y4","w eI4 n j ax n2 x aU4 m i:4 m A:3 w O:3 j aU1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 u:2 f A:4 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 W AA1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 W AY1 L ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 w ax n2 x w A:4 ts6 j ax n4 ts6_h y4","w eI4 n j ax n2 x aU4 m i:4 m A:3 w A:1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 W IH1 L ts6 j ax n4 ts6_h y4"],"retry": 2}' http://localhost:10180/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}
curl -v -X POST -H "Content-Type: application/json" -d @/home/hsiehpinghan/git/esun_ai_2022_summer/data/request.json http://35.194.149.173:10180/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}

# run gcp container with GPU
## GPU type: NVIDIA Tesla T4
## Machine type: n1-standard-4 (4 vCPU, 15 GB memory)
## enable docker gpu (https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/)
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
mkdir -p /tmp/redis/data
docker run --name redis \
  --restart=always \
  -p 6379:6379 \
  -v /tmp/redis/data:/data \
  -itd \
  redis:6.2.4 redis-server --appendonly yes
sudo mkdir -p /log/log_0
docker login
docker run --name esun_ai_2022_summer_0 \
  --gpus all \
  --restart=always \
  -e "TZ=Asia/Taipei" \
  -p 10180:5000 \
  -v /log/log_0:/tmp \
  --link redis:redis \
  -td hsiehpinghan/esun_ai_2022_summer:5.0.0
curl -v -X POST -H "Content-Type: application/json" -d '{"esun_uuid": "adefb7e8d9268b972b95b6fa53db93780b6b22fbf","esun_timestamp": 1590493849,"sentence_list": ["喂 你好 密碼 我 要 進去","喂 你好 密碼 哇 進去","喂 你好 密碼 的 話 進去","喂 您好 密碼 我 要 進去","喂 你好 密碼 無法 進去","喂 你好 密碼 waa 進去","喂 你好 密碼 while 進去","喂 你好 密碼 文化 進去","喂 你好 密碼 挖 進去","喂 您好 密碼 哇 進去"],"phoneme_sequence_list": ["w eI4 n i:3 x aU4 m i:4 m A:3 w O:3 j aU1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 w A:1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 t ax5 x w A:4 ts6 j ax n4 ts6_h y4","w eI4 n j ax n2 x aU4 m i:4 m A:3 w O:3 j aU1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 u:2 f A:4 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 W AA1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 W AY1 L ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 w ax n2 x w A:4 ts6 j ax n4 ts6_h y4","w eI4 n j ax n2 x aU4 m i:4 m A:3 w A:1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 W IH1 L ts6 j ax n4 ts6_h y4"],"retry": 2}' http://localhost:10180/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}
curl -v -X POST -H "Content-Type: application/json" -d @/home/hsiehpinghan/git/esun_ai_2022_summer/data/request.json http://35.194.149.173:10180/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}


# run local api container
mkdir -p /tmp/redis/data
docker run --name redis \
  --restart=always \
  -p 6379:6379 \
  -v /tmp/redis/data:/data \
  -itd \
  redis:6.2.4 redis-server --appendonly yes
docker run --rm -d \
  --link redis:redis \
  -e REDIS_1_HOST=redis \
  -e REDIS_1_NAME=redis \
  -p 18080:80 \
  erikdubbelboer/phpredisadmin:v1.13.2
http://localhost:18080/
mkdir -p /tmp/log_0
docker run --name esun_ai_2022_summer_0 \
  --restart=always \
  -e "TZ=Asia/Taipei" \
  -p 10180:5000 \
  -v /tmp/log_0:/log \
  --link redis:redis \
  -td hsiehpinghan/esun_ai_2022_summer:5.0.0
curl -v -X POST -H "Content-Type: application/json" -d '{"esun_uuid": "165e4f10a0b099c44791e901efb7e8d9268b972b95b6fa53db93780b6b22fbf","esun_timestamp": 1590493849,"sentence_list": ["喂 你好 密碼 我 要 進去","喂 你好 密碼 哇 進去","喂 你好 密碼 的 話 進去","喂 您好 密碼 我 要 進去","喂 你好 密碼 無法 進去","喂 你好 密碼 waa 進去","喂 你好 密碼 while 進去","喂 你好 密碼 文化 進去","喂 你好 密碼 挖 進去","喂 您好 密碼 哇 進去"],"phoneme_sequence_list": ["w eI4 n i:3 x aU4 m i:4 m A:3 w O:3 j aU1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 w A:1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 t ax5 x w A:4 ts6 j ax n4 ts6_h y4","w eI4 n j ax n2 x aU4 m i:4 m A:3 w O:3 j aU1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 u:2 f A:4 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 W AA1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 W AY1 L ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 w ax n2 x w A:4 ts6 j ax n4 ts6_h y4","w eI4 n j ax n2 x aU4 m i:4 m A:3 w A:1 ts6 j ax n4 ts6_h y4","w eI4 n i:3 x aU4 m i:4 m A:3 W IH1 L ts6 j ax n4 ts6_h y4"],"retry":2}' http://localhost:10180/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}
mkdir -p /tmp/log_1
docker run --name esun_ai_2022_summer_1 \
  --restart=always \
  -e "TZ=Asia/Taipei" \
  -p 10181:5000 \
  -v /tmp/log_1:/log \
  --link redis:redis \
  -td hsiehpinghan/esun_ai_2022_summer:5.0.0
curl -v -X POST -H "Content-Type: application/json" -d @/home/hsiehpinghan/git/esun_ai_2022_summer/data/request.json http://localhost:10181/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}
mkdir -p /tmp/log_2
docker run --name esun_ai_2022_summer_2 \
  --restart=always \
  -e "TZ=Asia/Taipei" \
  -p 10182:5000 \
  -v /tmp/log_2:/log \
  --link redis:redis \
  -td hsiehpinghan/esun_ai_2022_summer:5.0.0
curl -v -X POST -H "Content-Type: application/json" -d @/home/hsiehpinghan/git/esun_ai_2022_summer/data/request.json http://localhost:10182/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}
mkdir -p /tmp/log_3
docker run --name esun_ai_2022_summer_3 \
  --restart=always \
  -e "TZ=Asia/Taipei" \
  -p 10183:5000 \
  -v /tmp/log_3:/log \
  --link redis:redis \
  -td hsiehpinghan/esun_ai_2022_summer:5.0.0
curl -v -X POST -H "Content-Type: application/json" -d @/home/hsiehpinghan/git/esun_ai_2022_summer/data/request.json http://localhost:10183/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}
mkdir -p /tmp/log_4
docker run --name esun_ai_2022_summer_4 \
  --restart=always \
  -e "TZ=Asia/Taipei" \
  -p 10184:5000 \
  -v /tmp/log_4:/log \
  --link redis:redis \
  -td hsiehpinghan/esun_ai_2022_summer:5.0.0
curl -v -X POST -H "Content-Type: application/json" -d @/home/hsiehpinghan/git/esun_ai_2022_summer/data/request.json http://localhost:10184/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}
vi /root/nginx.conf
events {
    worker_connections  1024;
}
http {
  upstream my_upstream {
    server esun_ai_2022_summer_0:5000;
    server esun_ai_2022_summer_1:5000;
    server esun_ai_2022_summer_2:5000;
    server esun_ai_2022_summer_3:5000;
    server esun_ai_2022_summer_4:5000;
  }
  server {
    listen 11180;

    location / {
      proxy_pass http://my_upstream;
    }
  }
}
docker run --name load_balance \
  --restart=always \
  -d \
  -p 11180:11180 \
  --link esun_ai_2022_summer_0:esun_ai_2022_summer_0 \
  --link esun_ai_2022_summer_1:esun_ai_2022_summer_1 \
  --link esun_ai_2022_summer_2:esun_ai_2022_summer_2 \
  --link esun_ai_2022_summer_3:esun_ai_2022_summer_3 \
  --link esun_ai_2022_summer_4:esun_ai_2022_summer_4 \
  -v /root/nginx.conf:/etc/nginx/nginx.conf:ro \
  nginx:1.15.8-alpine
curl -v -X POST -H "Content-Type: application/json" -d @/home/hsiehpinghan/git/esun_ai_2022_summer/data/request.json http://localhost:11180/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}
curl -v -X POST -H "Content-Type: application/json" -d @/home/hsiehpinghan/git/esun_ai_2022_summer/data/request.json http://localhost:11180/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}
curl -v -X POST -H "Content-Type: application/json" -d @/home/hsiehpinghan/git/esun_ai_2022_summer/data/request.json http://localhost:11180/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}
curl -v -X POST -H "Content-Type: application/json" -d @/home/hsiehpinghan/git/esun_ai_2022_summer/data/request.json http://localhost:11180/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}
curl -v -X POST -H "Content-Type: application/json" -d @/home/hsiehpinghan/git/esun_ai_2022_summer/data/request.json http://localhost:11180/inference -w %{time_connect}:%{time_starttransfer}:%{time_total}

