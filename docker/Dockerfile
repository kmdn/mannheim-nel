FROM continuumio/anaconda

RUN apt-get update -y
RUN apt-get install software-properties-common -y python3-pip

RUN git clone https://github.com/rohitg1594/mannheim-nel
RUN git clone https://github.com/studio-ousia/ntee

RUN conda create -n py27 python=2.7
RUN conda create -n py36 python=3.6

WORKDIR mannheim-nel

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

