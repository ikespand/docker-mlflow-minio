FROM python:3.9-buster

LABEL maintainer="Sandeep Pandey <spandey.ike@gmail.com"
ARG PROTOBUF_VERSION=3.19.0
ARG MLFLOW_VERSION=1.19.0

RUN pip install --no-cache-dir protobuf==$PROTOBUF_VERSION mlflow==$MLFLOW_VERSION boto3