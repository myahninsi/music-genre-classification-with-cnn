FROM apache/airflow:2.8.0-python3.11

USER root
RUN apt-get update && apt-get install -y unzip && apt-get clean

COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

USER airflow
