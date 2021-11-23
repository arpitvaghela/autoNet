FROM nvidia/cuda:11.0-base

RUN apt-get update
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install python3.8 -y
RUN apt-get install python3.8-distutils -y

RUN apt-get install python-dev python3-dev python3.8-dev \
    build-essential libssl-dev libffi-dev \
    libxml2-dev libxslt1-dev zlib1g-dev -y

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc

RUN apt-get install curl wget -y

RUN apt-get install linux-libc-dev -y

RUN apt-get install libc6-dev -y
RUN apt-get install graphviz -y

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN  python3.8 get-pip.py
WORKDIR /usr/src/app

# dont write pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# dont buffer to stdout/stderr
ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /usr/src/app/requirements.txt

RUN pip3.8 install --no-cache-dir -r requirements.txt
