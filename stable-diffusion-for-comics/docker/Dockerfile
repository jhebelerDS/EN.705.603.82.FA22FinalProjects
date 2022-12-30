FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
WORKDIR /app