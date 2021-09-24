FROM python:3.8-slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN mkdir notepads
RUN mkdir data

ENTRYPOINT [ "/bin/sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/notepads --allow-root" ]
