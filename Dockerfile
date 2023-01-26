FROM python:3.8-slim

COPY . /ml-engineering

WORKDIR /ml-engineering

RUN python -m pip install --upgrade pip -U

ENTRYPOINT ["python", "simple_linear_regr.py"]
