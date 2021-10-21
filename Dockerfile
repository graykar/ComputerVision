FROM python:3.7-slim
FROM pytorch/pytorch


RUN apt-get update

WORKDIR /APP
COPY . ./

RUN mkdir pred_images

RUN mkdir processed_video

RUN pip install -r requirements.txt
