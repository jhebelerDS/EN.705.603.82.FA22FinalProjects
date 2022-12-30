FROM ubuntu:22.04
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
ENV DEBIAN_FRONTEND noninteractive
RUN apt install -y python3.9
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.9 /usr/bin/python3
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN apt-get install python3-pip python3.9-distutils ffmpeg -y
RUN apt install -y docker.io
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY code/ code
