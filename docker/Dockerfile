FROM continuumio/anaconda

RUN apt-get update -y
RUN apt-get install software-properties-common -y python3-pip

RUN git clone https://github.com/rohitg1594/mannheim-nel
RUN git clone https://github.com/studio-ousia/ntee

RUN pip install joblib
RUN mkdir data
RUN curl -O https://s3-ap-northeast-1.amazonaws.com/ntee/pub/models/ntee_300_sentence.joblib.gz
RUN gunzip ntee_300_sentence.joblib.gz
RUN python mannheim-nel/scripts/extract_embeddings.py ntee_300_sentence.joblib
RUN rm ntee_300_sentence.joblib

WORKDIR mannheim-nel

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

