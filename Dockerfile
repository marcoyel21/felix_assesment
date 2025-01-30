FROM python:3.8
ENV PYTHONUNBUFFERED 1
ENV TZ=America/Mexico_City
COPY . /
RUN apt update
RUN pip install --upgrade pip && pip install -r requirements.txt
