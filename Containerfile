FROM docker.io/civisanalytics/datascience-python
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt