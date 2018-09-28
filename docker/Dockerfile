FROM continuumio/anaconda

RUN apt-get update -y
RUN apt-get install software-properties-common -y python3-pip

RUN git clone https://github.com/rohitg1594/mannheim-nel
RUN git clone https://github.com/studio-ousia/ntee
RUN pip3 install --trusted-host pypi.python.org -r /mannheim-nel/requirements.txt

RUN pip install joblib
RUN pip install maris-trie

RUN mkdir data
RUN mkdir -p data/models
RUN mkdir -p data/embs

COPY ntee_300_sentence.joblib /data
COPY wiki-model.pickle /data/models

#RUN curl -O https://s3-ap-northeast-1.amazonaws.com/ntee/pub/models/ntee_300_sentence.joblib.gz
#RUN gunzip ntee_300_sentence.joblib.gz

RUN python /mannheim-nel/docker/scripts/extract_embeddings.py /ntee_300_sentence.joblib

RUN rm /data/ntee_300_sentence.joblib
RUN ls /data
RUN ls /data/embs/
RUN ls /data/models/
RUN ls /




