FROM ubuntu:18.04

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

RUN apt-get install -y python3 python3-pip && pip3 install --upgrade pip

WORKDIR /app
COPY . /app

RUN pip3 --no-cache-dir --use-feature=2020-resolver install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3", "api.py"]
